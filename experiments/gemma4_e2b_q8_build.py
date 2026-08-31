#!/usr/bin/env python3
"""Prepare and sweep provenance-bound direct-Q8 Gemma 4 E2B edits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from gguf import GGUFReader
from safetensors import safe_open
from safetensors.torch import load_file, save_file
import torch
from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM

from heretic_nx.edits import (
    GGUFQuantizedAblationPlan,
    GGUFQuantizedTensorEdit,
    GGUFStrengthSweepCandidate,
    apply_quantized_gguf_strength_sweep,
)
from heretic_nx.geometry.contrastive import fit_contrastive_axis
from heretic_nx.hashing import canonical_json, sha256_file, sha256_json
from heretic_nx.model import discover_structural_frontend


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "checkpoints" / "gemma4-e2b-it"
SOURCE_WEIGHTS = SOURCE / "model.safetensors"
BASE_Q8 = ROOT / "outputs" / "gemma-4-E2B-it-Q8_0.gguf"
OLD_RUN_DIR = ROOT / "runs" / "gemma4-e2b-residual-stream-prime"
ACTIVATIONS = OLD_RUN_DIR / "local-site-outputs.safetensors"
DETECTORS = OLD_RUN_DIR / "teacher-delta-distilled-detectors.safetensors"
DETECTOR_DIAGNOSTICS = OLD_RUN_DIR / "teacher-delta-distilled-diagnostics.json"
RUN_DIR = ROOT / "runs" / "gemma4-e2b-q8"
FACTORS = RUN_DIR / "lambda100-factors.safetensors"
PREPARATION = RUN_DIR / "lambda100-preparation.json"

MODEL_ID = "google/gemma-4-E2B-it"
MODEL_REVISION = "3e22461f65e89153144f8adb70e3b8c2cc9845a7"
GOOD_REVISION = "02c6a92cfcf11bb0c387334f8146d149d65b587f"
BAD_REVISION = "01cead01398926d81f7c52bdb790ee8cf77ebba7"
SAFE_LAMBDA = 100.0
FOLDS = 3
EXPECTED_SELECTED = (
    "L26:ffn_out",
    "L15:ffn_out",
    "L23:ffn_out",
    "L16:attention_out",
    "L17:attention_out",
    "L25:ffn_out",
    "L30:attention_out",
)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json(value) + b"\n")


def _source_metadata() -> dict[str, str]:
    with safe_open(ACTIVATIONS, framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
    expected = {
        "model_revision": MODEL_REVISION,
        "good_revision": GOOD_REVISION,
        "bad_revision": BAD_REVISION,
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise RuntimeError("activation cache does not match the pinned protocol")
    with safe_open(DETECTORS, framework="pt", device="cpu") as handle:
        detector_metadata = handle.metadata() or {}
    if detector_metadata.get("schema_version") != "gemma4-e2b-distilled-detectors-v1":
        raise RuntimeError("unsupported distilled detector artifact")
    return metadata


def _load_structure() -> Any:
    model = Gemma4ForCausalLM.from_pretrained(
        SOURCE,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map={"": "mps"},
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
        key_mapping={r"^model\.language_model\.": "model."},
    ).eval()
    report = discover_structural_frontend(model)
    del model
    torch.mps.synchronize()
    torch.mps.empty_cache()
    if (
        report.decoder_stack_path != "model.layers"
        or report.stream_dim != 1536
        or report.layer_count != 35
        or len(report.activation_sites) != 70
        or len(report.editable_targets) != 70
    ):
        raise RuntimeError("the pinned checkpoint has an unexpected structure")
    return report


def _site_id(target: Any) -> str:
    suffix = "attention_out" if target.role == "attention_output" else "ffn_out"
    return f"L{target.layer:02d}:{suffix}"


def _tensor_name(target: Any) -> str:
    suffix = "attn_output.weight" if target.role == "attention_output" else "ffn_down.weight"
    return f"blk.{target.layer}.{suffix}"


def prepare() -> dict[str, Any]:
    for path in (
        SOURCE_WEIGHTS,
        BASE_Q8,
        ACTIVATIONS,
        DETECTORS,
        DETECTOR_DIAGNOSTICS,
    ):
        if not path.is_file():
            raise RuntimeError(f"missing required artifact: {path}")
    cache_metadata = _source_metadata()
    report = _load_structure()
    targets = {
        _site_id(target): target
        for target in report.editable_targets
        if target.role in {"attention_output", "ffn_output"}
    }
    if len(targets) != 70:
        raise RuntimeError("structural target mapping is not one-to-one")

    activation_payload = load_file(ACTIVATIONS)
    ranking: list[dict[str, Any]] = []
    axes: dict[str, torch.Tensor] = {}
    for site_id, target in targets.items():
        safe = activation_payload[f"safe.{site_id}"].float()
        harmful = activation_payload[f"target.{site_id}"].float()
        evidence = fit_contrastive_axis(
            safe,
            harmful,
            folds=FOLDS,
            remove_safe_mean=False,
        )
        axis = evidence.axis.float().cpu().contiguous()
        safe_delta = (safe @ axis)[:, None] * axis[None, :]
        harmful_delta = (harmful @ axis)[:, None] * axis[None, :]
        safe_drift = float(
            torch.linalg.vector_norm(safe_delta)
            / torch.linalg.vector_norm(safe).clamp_min(1e-8)
        )
        harmful_drift = float(
            torch.linalg.vector_norm(harmful_delta)
            / torch.linalg.vector_norm(harmful).clamp_min(1e-8)
        )
        harmful_effect = float(torch.abs((harmful.mean(0) - safe.mean(0)) @ axis))
        score = (
            harmful_effect
            * max(harmful_drift, 1e-8)
            * max(evidence.fold_cosine_minimum, 1e-3)
            / max(safe_drift, 1e-6)
        )
        axes[site_id] = axis
        ranking.append(
            {
                "site_id": site_id,
                "layer": target.layer,
                "role": target.role,
                "module_path": target.module_path,
                "parameter_path": target.parameter_path,
                "tensor_name": _tensor_name(target),
                "input_dim": target.input_dim,
                "output_dim": target.output_dim,
                "score": score,
                "safe_relative_drift": safe_drift,
                "harmful_relative_drift": harmful_drift,
                "harmful_effect": harmful_effect,
                "fold_cosine_minimum": evidence.fold_cosine_minimum,
            }
        )
    ranking.sort(key=lambda row: (-float(row["score"]), str(row["site_id"])))
    selected = ranking[: len(EXPECTED_SELECTED)]
    selected_ids = tuple(str(row["site_id"]) for row in selected)
    if selected_ids != EXPECTED_SELECTED:
        raise RuntimeError(f"site ranking changed: {selected_ids}")

    detectors = load_file(DETECTORS)
    gguf_tensors = {tensor.name: tensor for tensor in GGUFReader(BASE_Q8).tensors}
    factor_payload: dict[str, torch.Tensor] = {}
    selected_payload: list[dict[str, Any]] = []
    for index, row in enumerate(selected):
        site_id = str(row["site_id"])
        tensor_name = str(row["tensor_name"])
        tensor = gguf_tensors.get(tensor_name)
        if tensor is None or tensor.tensor_type.name != "Q8_0":
            raise RuntimeError(f"Q8 target is missing: {tensor_name}")
        logical_shape = tuple(int(value) for value in reversed(tensor.shape.tolist()))
        if logical_shape != (int(row["output_dim"]), int(row["input_dim"])):
            raise RuntimeError(f"GGUF/HF shape disagreement for {tensor_name}")
        detector_key = f"lambda{SAFE_LAMBDA:g}.{site_id}"
        detector = detectors[detector_key].float().cpu().contiguous()
        if detector.shape != (int(row["input_dim"]),):
            raise RuntimeError(f"detector shape disagreement for {site_id}")
        a_key = f"site{index:02d}.axis"
        right_key = f"site{index:02d}.right"
        factor_payload[a_key] = axes[site_id][:, None].contiguous()
        factor_payload[right_key] = detector[:, None].contiguous()
        selected_payload.append({**row, "a_key": a_key, "right_key": right_key})

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    save_file(
        factor_payload,
        FACTORS,
        metadata={
            "schema_version": "gemma4-e2b-q8-lambda100-factors-v1",
            "model_revision": MODEL_REVISION,
            "structure_hash": report.structure_hash,
            "safe_lambda": f"{SAFE_LAMBDA:g}",
        },
    )
    preparation = {
        "schema_version": "gemma4-e2b-q8-preparation-v1",
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION},
        "source_weights": {
            "path": str(SOURCE_WEIGHTS),
            "sha256": sha256_file(SOURCE_WEIGHTS),
        },
        "base_q8": {"path": str(BASE_Q8), "sha256": sha256_file(BASE_Q8)},
        "llama_cpp_revision": "d7bd3bfcad3e29c7e49fd26f38c79ee3e9a3fd6b",
        "structure": {
            "decoder_stack_path": report.decoder_stack_path,
            "stream_dim": report.stream_dim,
            "layer_count": report.layer_count,
            "activation_count": len(report.activation_sites),
            "editable_target_count": len(report.editable_targets),
            "structure_hash": report.structure_hash,
        },
        "activation_cache": {
            "path": str(ACTIVATIONS),
            "sha256": sha256_file(ACTIVATIONS),
            "metadata": cache_metadata,
        },
        "detectors": {
            "path": str(DETECTORS),
            "sha256": sha256_file(DETECTORS),
            "diagnostics": str(DETECTOR_DIAGNOSTICS),
            "diagnostics_sha256": sha256_file(DETECTOR_DIAGNOSTICS),
            "safe_lambda": SAFE_LAMBDA,
        },
        "factor_artifact": {"path": str(FACTORS), "sha256": sha256_file(FACTORS)},
        "selected": selected_payload,
        "selected_sha256": sha256_json(selected_payload),
        "ranking": ranking,
    }
    write_json(PREPARATION, preparation)
    return preparation


def _verified_preparation() -> dict[str, Any]:
    if not PREPARATION.is_file() or not FACTORS.is_file():
        raise RuntimeError("run the prepare command first")
    preparation = json.loads(PREPARATION.read_text(encoding="utf-8"))
    if (
        preparation.get("schema_version") != "gemma4-e2b-q8-preparation-v1"
        or preparation.get("base_q8", {}).get("sha256") != sha256_file(BASE_Q8)
        or preparation.get("factor_artifact", {}).get("sha256") != sha256_file(FACTORS)
        or tuple(row["site_id"] for row in preparation.get("selected", []))
        != EXPECTED_SELECTED
    ):
        raise RuntimeError("preparation evidence is stale")
    return preparation


def plan(beta: float) -> Path:
    if beta <= 0:
        raise ValueError("beta must be positive")
    preparation = _verified_preparation()
    edits = tuple(
        GGUFQuantizedTensorEdit(
            tensor_name=str(row["tensor_name"]),
            expected_quantization="Q8_0",
            a_key=str(row["a_key"]),
            right_key=str(row["right_key"]),
            strength=beta,
            preserve_row_norms=False,
            preserve_original_blocks=True,
            quantization_multipliers=(1.0,),
            minimum_block_improvement=0.0,
            require_payload_change=True,
        )
        for row in preparation["selected"]
    )
    result = GGUFQuantizedAblationPlan(
        source_sha256=str(preparation["base_q8"]["sha256"]),
        tensor_artifact_sha256=str(preparation["factor_artifact"]["sha256"]),
        edits=edits,
        row_chunk_size=256,
        verify_untouched_bytes=True,
    )
    label = str(beta).replace(".", "p")
    path = RUN_DIR / f"lambda100-beta{label}.plan.json"
    result.write(path)
    return path


def sweep(betas: list[float]) -> dict[str, Any]:
    if len(betas) < 2 or len(betas) != len(set(betas)):
        raise ValueError("sweep requires at least two unique beta values")
    candidates = []
    for beta in betas:
        label = f"l100-b{str(beta).replace('.', 'p')}"
        candidates.append(
            GGUFStrengthSweepCandidate(
                label=label,
                plan_path=plan(beta),
                output_path=ROOT / "outputs" / f"gemma4-e2b-q8-{label}.gguf",
            )
        )
    result = apply_quantized_gguf_strength_sweep(BASE_Q8, FACTORS, candidates)
    sweep_label = "-".join(str(beta).replace(".", "p") for beta in betas)
    report_path = RUN_DIR / f"lambda100-strength-sweep-{sweep_label}.json"
    write_json(report_path, result)
    print(json.dumps({"report": str(report_path), **result}, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare")
    sweep_parser = subparsers.add_parser("sweep")
    sweep_parser.add_argument("--betas", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        print(json.dumps(prepare(), indent=2), flush=True)
    else:
        betas = [float(value) for value in args.betas.split(",")]
        sweep(betas)


if __name__ == "__main__":
    main()
