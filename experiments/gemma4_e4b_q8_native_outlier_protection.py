#!/usr/bin/env python3
"""Protect E4B benign KL outliers in the native Q8 activation geometry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from safetensors.torch import load_file, save_file
import torch

from heretic_nx.edits import (
    GGUFQuantizedAblationPlan,
    GGUFQuantizedTensorEdit,
    apply_quantized_gguf_ablation,
)
from heretic_nx.hashing import canonical_json, sha256_file

import gemma4_e4b_q8_build as parent


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "runs" / "gemma4-e4b-q8"
INPUT_DIR = RUN_DIR / "q8-protected-inputs"
BASE_STRENGTHS = (14.0, 6.0, 8.0, 8.0, 8.0, 14.0, 8.0)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json(value) + b"\n")


def native_inputs(
    input_dir: Path, row_count: int, site: int, expected_dim: int,
    all_positions: bool = False,
) -> np.ndarray:
    columns = []
    for row in range(row_count):
        path = input_dir / f"row{row}.site{site}.f32"
        value = np.fromfile(path, dtype="<f4")
        valid_size = (value.size > 0 and value.size % expected_dim == 0) if all_positions else value.shape == (expected_dim,)
        if not valid_size or not np.isfinite(value).all():
            raise RuntimeError(f"invalid native activation: {path}/{value.shape}")
        columns.append(value.reshape(-1, expected_dim).T)
    return np.concatenate(columns, axis=1)


def build(
    scale: float,
    input_dir: Path,
    row_count: int,
    requantize_all: bool,
    multipliers: dict[int, float],
    all_positions: bool = False,
) -> dict[str, Any]:
    if scale <= 0:
        raise ValueError("scale must be positive")
    if row_count < 1:
        raise ValueError("row_count must be positive")
    if any(index < 0 or index >= len(BASE_STRENGTHS) for index in multipliers):
        raise ValueError("multiplier site index is out of range")
    if any(value <= 0 for value in multipliers.values()):
        raise ValueError("site multipliers must be positive")
    preparation = parent.engine._verified_preparation()
    factors = load_file(parent.engine.FACTORS)
    payload: dict[str, torch.Tensor] = {}
    edits = []
    diagnostics = []
    for index, row in enumerate(preparation["selected"]):
        site = f"site{index:02d}"
        axis = factors[f"{site}.axis"].float().cpu()
        right = (
            BASE_STRENGTHS[index]
            * scale
            * multipliers.get(index, 1.0)
            * factors[f"{site}.right"].float().cpu()
        )
        inputs = native_inputs(input_dir, row_count, index, right.shape[0], all_positions).astype(
            np.float64
        )
        basis, singular, _ = np.linalg.svd(inputs, full_matrices=False)
        retained = singular > singular[0] * 1e-6
        basis = basis[:, retained]
        projected = right.double().numpy() - basis @ (basis.T @ right.double().numpy())
        projected = torch.from_numpy(projected.astype(np.float32))
        axis_key = f"{site}.axis"
        right_key = f"{site}.native_right"
        payload[axis_key] = axis.contiguous()
        payload[right_key] = projected.contiguous()
        edits.append(
            GGUFQuantizedTensorEdit(
                tensor_name=str(row["tensor_name"]),
                expected_quantization="Q8_0",
                a_key=axis_key,
                right_key=right_key,
                strength=1.0,
                preserve_row_norms=False,
                preserve_original_blocks=not requantize_all,
                quantization_multipliers=(1.0,),
                minimum_block_improvement=0.0,
                require_payload_change=True,
            )
        )
        diagnostics.append(
            {
                "site_id": str(row["site_id"]),
                "tensor_name": str(row["tensor_name"]),
                "strength": (
                    BASE_STRENGTHS[index]
                    * scale
                    * multipliers.get(index, 1.0)
                ),
                "native_input_rank": int(retained.sum()),
                "native_input_columns": int(inputs.shape[1]),
                "protected_dot_before_l2": float(
                    np.linalg.norm(inputs.T @ right.double().numpy())
                ),
                "protected_dot_after_l2": float(
                    np.linalg.norm(inputs.T @ projected.double().numpy())
                ),
            }
        )

    scale_label = f"{scale:g}".replace(".", "p")
    block_label = "-allblocks" if requantize_all else ""
    position_label = "-allpositions" if all_positions else ""
    multiplier_label = "".join(
        f"-m{index}x{value:g}".replace(".", "p")
        for index, value in sorted(multipliers.items())
    )
    tag = (
        f"coord-nearmiss-native-b{row_count}-protected-s{scale_label}"
        f"{multiplier_label}"
        f"{block_label}"
        f"{position_label}"
    )
    factor_path = RUN_DIR / f"{tag}-factors.safetensors"
    plan_path = RUN_DIR / f"{tag}.plan.json"
    output_path = ROOT / "outputs" / f"gemma4-e4b-q8-{tag}.gguf"
    report_path = RUN_DIR / f"{tag}.build.json"
    input_files = sorted(input_dir.glob("*.f32"))
    save_file(
        payload,
        factor_path,
        metadata={
            "schema_version": "gemma4-e4b-q8-native-outlier-protection-v1",
            "base_sha256": sha256_file(parent.engine.BASE_Q8),
            "source_factors_sha256": sha256_file(parent.engine.FACTORS),
            "native_input_sha256": ",".join(sha256_file(path) for path in input_files),
        },
    )
    plan = GGUFQuantizedAblationPlan(
        source_sha256=sha256_file(parent.engine.BASE_Q8),
        tensor_artifact_sha256=sha256_file(factor_path),
        edits=tuple(edits),
        row_chunk_size=256,
        verify_untouched_bytes=True,
    )
    plan.write(plan_path)
    merge = apply_quantized_gguf_ablation(
        parent.engine.BASE_Q8, output_path, plan_path, factor_path
    )
    result = {
        "schema_version": "gemma4-e4b-q8-native-outlier-protected-build-v1",
        "scale": scale,
        "all_positions": all_positions,
        "multipliers": {str(index): value for index, value in multipliers.items()},
        "requantize_all": requantize_all,
        "native_inputs": [
            {"path": str(path), "sha256": sha256_file(path)} for path in input_files
        ],
        "factor_artifact": {"path": str(factor_path), "sha256": sha256_file(factor_path)},
        "diagnostics": diagnostics,
        "merge": merge,
    }
    write_json(report_path, result)
    return {"report": str(report_path), **merge}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=float, default=1.1)
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    parser.add_argument("--row-count", type=int, default=2)
    parser.add_argument("--requantize-all", action="store_true")
    parser.add_argument("--multiplier", action="append", default=[])
    parser.add_argument("--all-positions", action="store_true")
    args = parser.parse_args()
    multipliers = {}
    for raw in args.multiplier:
        site, separator, value = raw.partition("=")
        if not separator or int(site) in multipliers:
            raise ValueError(f"invalid or duplicate multiplier: {raw}")
        multipliers[int(site)] = float(value)
    print(
        json.dumps(
            build(
                args.scale,
                args.input_dir.resolve(strict=True),
                args.row_count,
                args.requantize_all,
                multipliers,
                args.all_positions,
            ),
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
