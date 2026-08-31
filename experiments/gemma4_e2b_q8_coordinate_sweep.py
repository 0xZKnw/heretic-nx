#!/usr/bin/env python3
"""Shared-dequantization one-site coordinate search around the best Gemma Q8 profile."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from heretic_nx.edits import (
    GGUFQuantizedAblationPlan,
    GGUFQuantizedTensorEdit,
    GGUFStrengthSweepCandidate,
    apply_quantized_gguf_ablation,
    apply_quantized_gguf_strength_sweep,
)
from heretic_nx.hashing import canonical_json, sha256_file

import gemma4_e2b_q8_build as parent


FACTORS = parent.RUN_DIR / "l100-b4p0-repair-g1p875.safetensors"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json(value) + b"\n")


def make_plan(
    *,
    selected_index: int,
    multiplier: float,
    verify_untouched: bool,
    tag: str,
    fixed_strengths: dict[int, float] | None = None,
) -> Path:
    preparation = parent._verified_preparation()
    if not 0 <= selected_index < len(preparation["selected"]):
        raise ValueError("selected index is out of range")
    if multiplier <= 0:
        raise ValueError("multiplier must be positive")
    fixed = dict(fixed_strengths or {})
    if selected_index in fixed:
        raise ValueError("selected coordinate cannot also be fixed")
    if any(
        not 0 <= index < len(preparation["selected"]) or value <= 0
        for index, value in fixed.items()
    ):
        raise ValueError("fixed strengths must have valid indices and positive values")
    edits = []
    for index, row in enumerate(preparation["selected"]):
        strength = multiplier if index == selected_index else fixed.get(index, 1.0)
        edits.append(
            GGUFQuantizedTensorEdit(
                tensor_name=str(row["tensor_name"]),
                expected_quantization="Q8_0",
                a_key=f"site{index:02d}.axis",
                right_key=f"site{index:02d}.composed_right",
                strength=strength,
                preserve_row_norms=False,
                preserve_original_blocks=True,
                quantization_multipliers=(1.0,),
                minimum_block_improvement=0.0,
                require_payload_change=True,
            )
        )
    plan = GGUFQuantizedAblationPlan(
        source_sha256=sha256_file(parent.BASE_Q8),
        tensor_artifact_sha256=sha256_file(FACTORS),
        edits=tuple(edits),
        row_chunk_size=256,
        verify_untouched_bytes=verify_untouched,
    )
    path = parent.RUN_DIR / f"{tag}.plan.json"
    plan.write(path)
    return path


def search(
    multipliers: list[float],
    site_indices: list[int] | None = None,
    fixed_strengths: dict[int, float] | None = None,
) -> dict[str, Any]:
    preparation = parent._verified_preparation()
    selected_indices = site_indices or list(range(len(preparation["selected"])))
    if len(selected_indices) != len(set(selected_indices)) or any(
        not 0 <= index < len(preparation["selected"]) for index in selected_indices
    ):
        raise ValueError("site indices must be unique and in range")
    fixed = dict(fixed_strengths or {})
    if set(selected_indices) & set(fixed):
        raise ValueError("searched and fixed site indices must be disjoint")
    fixed_label = "".join(
        f"-f{index:02d}m{str(value).replace('.', 'p')}"
        for index, value in sorted(fixed.items())
    )
    candidates = []
    profiles = []
    for index in selected_indices:
        row = preparation["selected"][index]
        for multiplier in multipliers:
            encoded = str(multiplier).replace(".", "p")
            tag = f"coord{fixed_label}-s{index:02d}-m{encoded}"
            plan = make_plan(
                selected_index=index,
                multiplier=multiplier,
                verify_untouched=False,
                tag=tag,
                fixed_strengths=fixed,
            )
            output = parent.ROOT / "outputs" / f"gemma4-e2b-q8-{tag}.gguf"
            candidates.append(
                GGUFStrengthSweepCandidate(
                    label=tag,
                    plan_path=plan,
                    output_path=output,
                )
            )
            profiles.append(
                {
                    "tag": tag,
                    "site_index": index,
                    "site_id": str(row["site_id"]),
                    "multiplier": multiplier,
                    "plan": str(plan),
                    "output": str(output),
                }
            )
    merge = apply_quantized_gguf_strength_sweep(
        parent.BASE_Q8,
        FACTORS,
        candidates,
        fast_search=True,
    )
    report = {
        "schema_version": "gemma4-e2b-q8-coordinate-search-v1",
        "parent_profile": {"beta": 4.0, "repair_gamma": 1.875},
        "fixed_strengths": fixed,
        "factor_artifact": {"path": str(FACTORS), "sha256": sha256_file(FACTORS)},
        "profiles": profiles,
        "merge": merge,
    }
    index_label = "-".join(f"s{index:02d}" for index in selected_indices)
    multiplier_label = "-".join(str(value).replace(".", "p") for value in multipliers)
    path = parent.RUN_DIR / f"coordinate-search-{index_label}-{multiplier_label}.json"
    write_json(path, report)
    return {"report": str(path), "candidate_count": len(candidates), **merge}


def exact(
    selected_index: int,
    multiplier: float,
    tag: str,
    fixed_strengths: dict[int, float] | None = None,
) -> dict[str, Any]:
    plan = make_plan(
        selected_index=selected_index,
        multiplier=multiplier,
        verify_untouched=True,
        tag=tag,
        fixed_strengths=fixed_strengths,
    )
    output = parent.ROOT / "outputs" / f"gemma4-e2b-q8-{tag}.gguf"
    merge = apply_quantized_gguf_ablation(
        parent.BASE_Q8,
        output,
        plan,
        FACTORS,
    )
    report = {
        "schema_version": "gemma4-e2b-q8-coordinate-final-v1",
        "parent_profile": {"beta": 4.0, "repair_gamma": 1.875},
        "selected_index": selected_index,
        "multiplier": multiplier,
        "fixed_strengths": dict(fixed_strengths or {}),
        "factor_artifact": {"path": str(FACTORS), "sha256": sha256_file(FACTORS)},
        "merge": merge,
    }
    path = parent.RUN_DIR / f"{tag}.build.json"
    write_json(path, report)
    return {"report": str(path), **merge}


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    search_parser = subparsers.add_parser("search")
    search_parser.add_argument("--multipliers", default="0.5,1.5")
    search_parser.add_argument("--site-indices")
    search_parser.add_argument("--fixed-strength", action="append", default=[])
    exact_parser = subparsers.add_parser("exact")
    exact_parser.add_argument("--site-index", type=int, required=True)
    exact_parser.add_argument("--multiplier", type=float, required=True)
    exact_parser.add_argument("--tag", required=True)
    exact_parser.add_argument("--fixed-strength", action="append", default=[])
    args = parser.parse_args()
    fixed_strengths: dict[int, float] = {}
    for raw in args.fixed_strength:
        index, separator, value = raw.partition("=")
        if not separator or int(index) in fixed_strengths:
            raise ValueError(f"invalid or duplicate --fixed-strength: {raw}")
        fixed_strengths[int(index)] = float(value)
    if args.command == "search":
        site_indices = (
            [int(value) for value in args.site_indices.split(",")]
            if args.site_indices
            else None
        )
        result = search(
            [float(value) for value in args.multipliers.split(",")],
            site_indices,
            fixed_strengths,
        )
    else:
        result = exact(
            args.site_index,
            args.multiplier,
            args.tag,
            fixed_strengths,
        )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
