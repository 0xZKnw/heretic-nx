#!/usr/bin/env python3
"""Protect the single benign KL outlier from the selected Gemma Q8 edit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import load_dataset
from safetensors.torch import load_file, save_file
import torch
from transformers import AutoTokenizer
from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM

from heretic_nx.edits import (
    GGUFQuantizedAblationPlan,
    GGUFQuantizedTensorEdit,
    apply_quantized_gguf_ablation,
)
from heretic_nx.hashing import canonical_json, sha256_file, sha256_json

import gemma4_e2b_q8_build as parent
import gemma4_e2b_q8_eval as refusal
import gemma4_e2b_q8_kl as kl


INPUTS = parent.RUN_DIR / "benign-row53-inputs.safetensors"
SELECTED_FACTORS = parent.RUN_DIR / "l100-b4p0-repair-g1p875.safetensors"
PROTECTED_FACTORS = parent.RUN_DIR / "final-row53-protected-factors.safetensors"
PLAN = parent.RUN_DIR / "final-row53-protected.plan.json"
OUTPUT = parent.ROOT / "outputs" / "gemma4-e2b-q8-final-row53-protected.gguf"
REPORT = parent.RUN_DIR / "final-row53-protected.build.json"
ROW_INDEX = 52
STRENGTHS = {1: 1.5, 4: 1.25}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json(value) + b"\n")


@torch.inference_mode()
def capture() -> dict[str, Any]:
    preparation = parent._verified_preparation()
    tokenizer = AutoTokenizer.from_pretrained(
        refusal.TOKENIZER_PATH, local_files_only=True
    )
    rows = load_dataset(kl.GOOD_DATASET, revision=kl.GOOD_REVISION, split="test")
    prompt = str(rows[ROW_INDEX]["text"])
    rendered = refusal.render(tokenizer, [prompt])[0]
    tokens = tokenizer.encode(rendered, add_special_tokens=False)
    model = Gemma4ForCausalLM.from_pretrained(
        parent.SOURCE,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map={"": "mps"},
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
        key_mapping={r"^model\.language_model\.": "model."},
    ).eval()
    captured: dict[str, torch.Tensor] = {}
    handles = []
    for index, row in enumerate(preparation["selected"]):
        module = model.get_submodule(str(row["module_path"]))

        def hook(_module: Any, inputs: tuple[Any, ...], *, key=f"site{index:02d}") -> None:
            captured[key] = (
                inputs[0][0, -1].detach().float().cpu().contiguous()
            )

        handles.append(module.register_forward_pre_hook(hook))
    try:
        encoded = tokenizer(
            rendered,
            add_special_tokens=False,
            return_tensors="pt",
        )
        encoded = {key: value.to("mps") for key, value in encoded.items()}
        model(**encoded, use_cache=False, return_dict=True)
    finally:
        for handle in handles:
            handle.remove()
    if len(captured) != len(preparation["selected"]):
        raise RuntimeError("did not capture every selected module input")
    save_file(
        captured,
        INPUTS,
        metadata={
            "schema_version": "gemma4-e2b-benign-row-inputs-v1",
            "model_revision": parent.MODEL_REVISION,
            "dataset_revision": kl.GOOD_REVISION,
            "row_index_zero_based": str(ROW_INDEX),
            "prompt_tokens_sha256": sha256_json(tokens),
        },
    )
    result = {
        "schema_version": "gemma4-e2b-benign-row-inputs-v1",
        "model": {"id": parent.MODEL_ID, "revision": parent.MODEL_REVISION},
        "dataset": {
            "id": kl.GOOD_DATASET,
            "revision": kl.GOOD_REVISION,
            "split": "test",
            "row_index_zero_based": ROW_INDEX,
            "row_index_one_based": ROW_INDEX + 1,
        },
        "prompt": prompt,
        "prompt_tokens_sha256": sha256_json(tokens),
        "artifact": {"path": str(INPUTS), "sha256": sha256_file(INPUTS)},
        "shapes": {key: list(value.shape) for key, value in captured.items()},
    }
    write_json(INPUTS.with_suffix(".json"), result)
    return result


def build() -> dict[str, Any]:
    preparation = parent._verified_preparation()
    if not INPUTS.is_file():
        raise RuntimeError("capture the protected benign input first")
    factors = load_file(SELECTED_FACTORS)
    inputs = load_file(INPUTS)
    payload: dict[str, torch.Tensor] = {}
    edits = []
    diagnostics = []
    for index, row in enumerate(preparation["selected"]):
        axis_key = f"site{index:02d}.axis"
        right_key = f"site{index:02d}.protected_right"
        right = (
            float(STRENGTHS.get(index, 1.0))
            * factors[f"site{index:02d}.composed_right"].float()
        ).cpu()
        protected_input = inputs[f"site{index:02d}"].float().cpu()
        denominator = protected_input.square().sum().clamp_min(1e-12)
        coefficient = (protected_input[:, None] * right).sum(dim=0) / denominator
        protected = right - protected_input[:, None] * coefficient[None, :]
        before = float((protected_input[:, None] * right).sum().abs())
        after = float((protected_input[:, None] * protected).sum().abs())
        payload[axis_key] = factors[axis_key].float().cpu().contiguous()
        payload[right_key] = protected.contiguous()
        edits.append(
            GGUFQuantizedTensorEdit(
                tensor_name=str(row["tensor_name"]),
                expected_quantization="Q8_0",
                a_key=axis_key,
                right_key=right_key,
                strength=1.0,
                preserve_row_norms=False,
                preserve_original_blocks=True,
                quantization_multipliers=(1.0,),
                minimum_block_improvement=0.0,
                require_payload_change=True,
            )
        )
        diagnostics.append(
            {
                "site_id": str(row["site_id"]),
                "tensor_name": str(row["tensor_name"]),
                "selected_strength": float(STRENGTHS.get(index, 1.0)),
                "protected_dot_before": before,
                "protected_dot_after": after,
                "protected_dot_reduction": after / max(before, 1e-30),
            }
        )
    save_file(
        payload,
        PROTECTED_FACTORS,
        metadata={
            "schema_version": "gemma4-e2b-q8-row53-protected-factors-v1",
            "model_revision": parent.MODEL_REVISION,
            "source_factors_sha256": sha256_file(SELECTED_FACTORS),
            "protected_inputs_sha256": sha256_file(INPUTS),
        },
    )
    plan = GGUFQuantizedAblationPlan(
        source_sha256=sha256_file(parent.BASE_Q8),
        tensor_artifact_sha256=sha256_file(PROTECTED_FACTORS),
        edits=tuple(edits),
        row_chunk_size=256,
        verify_untouched_bytes=True,
    )
    plan.write(PLAN)
    merge = apply_quantized_gguf_ablation(
        parent.BASE_Q8,
        OUTPUT,
        PLAN,
        PROTECTED_FACTORS,
    )
    result = {
        "schema_version": "gemma4-e2b-q8-row53-protected-build-v1",
        "protected_row": ROW_INDEX + 1,
        "source_profile": {
            "beta": 4.0,
            "repair_gamma": 1.875,
            "strengths": {str(key): value for key, value in STRENGTHS.items()},
        },
        "protected_inputs": {"path": str(INPUTS), "sha256": sha256_file(INPUTS)},
        "factor_artifact": {
            "path": str(PROTECTED_FACTORS),
            "sha256": sha256_file(PROTECTED_FACTORS),
        },
        "diagnostics": diagnostics,
        "merge": merge,
    }
    write_json(REPORT, result)
    return {"report": str(REPORT), **merge}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("capture", "build"))
    args = parser.parse_args()
    result = capture() if args.command == "capture" else build()
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
