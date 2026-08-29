#!/usr/bin/env python3
"""Collect prompt-boundary activations for Ling-3.0-tiny dense output sites.

The native BF16 checkpoint is used only as a research instrument. Candidate
weights are still produced by editing the pinned Q8 GGUF directly.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import time
from typing import Any

from datasets import load_dataset
from safetensors.torch import save_file
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.lfm25_2p6b_residual_stream import (
    BAD_DATASET,
    BAD_REVISION,
    GOOD_DATASET,
    GOOD_REVISION,
)
from experiments.ling3_tiny_q8_eval import render
from heretic_nx.hashing import canonical_json, sha256_json


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "checkpoints" / "ling3-tiny"
RUN_DIR = ROOT / "runs" / "ling3-tiny-q8-direct"
OUTPUT = RUN_DIR / "native-dense-site-outputs.safetensors"
REPORT = RUN_DIR / "native-dense-site-outputs.json"
HIDDEN_SIZE = 1536
LAYERS = 24


def dense_sites(model: Any) -> list[tuple[str, str, torch.nn.Module]]:
    sites = []
    for layer_index, layer in enumerate(model.model.layers):
        attention = layer.attention
        projection = attention.dense if hasattr(attention, "dense") else attention.o_proj
        sites.append(
            (
                f"L{layer_index:02d}:attention_out",
                f"blk.{layer_index}.attn_output.weight",
                projection,
            )
        )
        if layer_index == 0:
            down = layer.mlp.down_proj
            tensor_name = "blk.0.ffn_down.weight"
        else:
            down = layer.mlp.shared_experts.down_proj
            tensor_name = f"blk.{layer_index}.ffn_down_shexp.weight"
        sites.append((f"L{layer_index:02d}:dense_ffn_out", tensor_name, down))
    if len(sites) != 2 * LAYERS:
        raise RuntimeError(f"expected {2 * LAYERS} dense sites, found {len(sites)}")
    return sites


def encoded_prompts(tokenizer: Any, texts: list[str]) -> list[list[int]]:
    prompts = render(tokenizer, texts)
    return [
        [int(token) for token in row]
        for row in tokenizer(prompts, add_special_tokens=False)["input_ids"]
    ]


def collect(
    model: Any,
    token_rows: list[list[int]],
    sites: list[tuple[str, str, torch.nn.Module]],
    *,
    batch_size: int,
    label: str,
) -> torch.Tensor:
    output = torch.empty(len(token_rows), len(sites), HIDDEN_SIZE, dtype=torch.bfloat16)
    captured: dict[int, torch.Tensor] = {}
    handles = []
    for site_index, (_site_id, _tensor_name, module) in enumerate(sites):
        def hook(_module, _inputs, value, *, index=site_index):
            captured[index] = value[:, -1, :].detach().to(device="cpu", dtype=torch.bfloat16)

        handles.append(module.register_forward_hook(hook))

    grouped: dict[int, list[int]] = defaultdict(list)
    for row_index, tokens in enumerate(token_rows):
        grouped[len(tokens)].append(row_index)
    completed = 0
    started = time.time()
    try:
        with torch.inference_mode():
            for length in sorted(grouped):
                row_indices = grouped[length]
                for start in range(0, len(row_indices), batch_size):
                    indices = row_indices[start : start + batch_size]
                    input_ids = torch.tensor(
                        [token_rows[index] for index in indices],
                        dtype=torch.long,
                    )
                    captured.clear()
                    model.model(input_ids=input_ids, use_cache=False, return_dict=True)
                    if set(captured) != set(range(len(sites))):
                        raise RuntimeError(
                            f"dense hooks did not all fire: {len(captured)}/{len(sites)}"
                        )
                    for site_index in range(len(sites)):
                        output[indices, site_index] = captured[site_index]
                    completed += len(indices)
                    print(
                        json.dumps(
                            {
                                "native_dense": label,
                                "completed": completed,
                                "total": len(token_rows),
                                "token_length": length,
                                "seconds": round(time.time() - started, 3),
                            }
                        ),
                        flush=True,
                    )
    finally:
        for handle in handles:
            handle.remove()
    if not torch.isfinite(output.float()).all():
        raise RuntimeError(f"non-finite native outputs for {label}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--rows", type=int, default=104)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--report", type=Path, default=REPORT)
    args = parser.parse_args()
    if args.batch_size <= 0 or not 1 <= args.rows <= 104:
        raise ValueError("batch size must be positive and rows must be in 1..104")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    safe_rows = load_dataset(
        GOOD_DATASET,
        revision=GOOD_REVISION,
        split=f"train[:{args.rows}]",
    )
    target_all = load_dataset(BAD_DATASET, revision=BAD_REVISION, split="test")
    target_rows = target_all.select(range(args.rows))
    safe_tokens = encoded_prompts(tokenizer, [str(row["text"]) for row in safe_rows])
    target_tokens = encoded_prompts(tokenizer, [str(row["text"]) for row in target_rows])

    started = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map={"": "cpu"},
        low_cpu_mem_usage=True,
        attn_implementation="eager",
        force_download=True,
    ).eval()
    sites = dense_sites(model)
    loaded_seconds = time.time() - started
    safe = collect(
        model,
        safe_tokens,
        sites,
        batch_size=args.batch_size,
        label="safe",
    )
    target = collect(
        model,
        target_tokens,
        sites,
        batch_size=args.batch_size,
        label="target",
    )
    manifest = {
        "schema_version": "ling3-tiny-native-dense-sites-v1",
        "research_model": "inclusionAI/Ling-3.0-tiny",
        "research_revision": "b61f4338de3e68ffc9c0bc1ed5e902981a4a929e",
        "candidate_weight_domain": "pinned Q8 GGUF only",
        "safe_dataset": [GOOD_DATASET, GOOD_REVISION],
        "target_dataset": [BAD_DATASET, BAD_REVISION],
        "rows": args.rows,
        "thinking": "off",
        "safe_prompt_tokens_sha256": sha256_json(safe_tokens),
        "target_prompt_tokens_sha256": sha256_json(target_tokens),
        "site_ids": [site_id for site_id, _tensor_name, _module in sites],
        "tensor_names": [tensor_name for _site_id, tensor_name, _module in sites],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {"safe": safe.contiguous(), "target": target.contiguous()},
        args.output,
        metadata={"manifest_sha256": sha256_json(manifest)},
    )
    report = {
        **manifest,
        "shape": list(safe.shape),
        "dtype": str(safe.dtype),
        "model_load_seconds": loaded_seconds,
        "total_seconds": time.time() - started,
        "output": str(args.output.resolve()),
    }
    args.report.write_bytes(canonical_json(report) + b"\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
