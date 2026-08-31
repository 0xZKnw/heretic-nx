#!/usr/bin/env python3
"""Materialize provenance-bound additive-repair candidates in native Q8_0."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from safetensors.torch import load_file, save_file
import torch

from heretic_nx.edits import (
    GGUFQuantizedAblationPlan,
    GGUFQuantizedTensorEdit,
    apply_quantized_gguf_ablation,
)
from heretic_nx.hashing import canonical_json, sha256_file, sha256_json

import gemma4_e2b_q8_build as parent


ROOT = parent.ROOT
RUN_DIR = parent.RUN_DIR
REPAIR_DETECTORS = (
    parent.OLD_RUN_DIR / "teacher-delta-additive-repair-detectors.safetensors"
)
REPAIR_DIAGNOSTICS = (
    parent.OLD_RUN_DIR / "teacher-delta-additive-repair-diagnostics.json"
)
DEFAULT_PARENT_BETA = 3.0


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json(value) + b"\n")


def label_number(value: float) -> str:
    return str(value).replace(".", "p")


def build(
    parent_beta: float,
    gamma: float,
    *,
    site_gammas: dict[str, float] | None = None,
    fast_search: bool = False,
    tag: str | None = None,
) -> dict[str, Any]:
    if parent_beta <= 0 or gamma <= 0:
        raise ValueError("parent beta and gamma must be positive")
    preparation = parent._verified_preparation()
    overrides = dict(site_gammas or {})
    selected_ids = {str(row["site_id"]) for row in preparation["selected"]}
    unknown = sorted(set(overrides) - selected_ids)
    if unknown:
        raise ValueError(f"unknown site gamma overrides: {unknown}")
    if any(value <= 0 for value in overrides.values()):
        raise ValueError("site gamma overrides must be positive")
    for path in (REPAIR_DETECTORS, REPAIR_DIAGNOSTICS):
        if not path.is_file():
            raise RuntimeError(f"missing repair evidence: {path}")

    parent_factors = load_file(parent.FACTORS)
    repair_detectors = load_file(REPAIR_DETECTORS)
    payload: dict[str, torch.Tensor] = {}
    edits = []
    composition = []
    for index, row in enumerate(preparation["selected"]):
        site_id = str(row["site_id"])
        axis_key = f"site{index:02d}.axis"
        parent_key = f"site{index:02d}.right"
        right_key = f"site{index:02d}.composed_right"
        axis = parent_factors[axis_key].float().cpu().contiguous()
        parent_right = parent_factors[parent_key].float().cpu().contiguous()
        repair = repair_detectors[site_id].float().cpu()[:, None].contiguous()
        if parent_right.shape != repair.shape:
            raise RuntimeError(f"repair detector shape mismatch for {site_id}")
        site_gamma = overrides.get(site_id, gamma)
        composed = parent_beta * parent_right + site_gamma * repair
        payload[axis_key] = axis
        payload[right_key] = composed.contiguous()
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
        composition.append(
            {
                "site_id": site_id,
                "tensor_name": str(row["tensor_name"]),
                "parent_beta": parent_beta,
                "repair_gamma": site_gamma,
            }
        )

    if tag is None:
        tag = f"l100-b{label_number(parent_beta)}-repair-g{label_number(gamma)}"
    factors = RUN_DIR / f"{tag}.safetensors"
    save_file(
        payload,
        factors,
        metadata={
            "schema_version": "gemma4-e2b-q8-additive-repair-factors-v1",
            "model_revision": parent.MODEL_REVISION,
            "parent_beta": f"{parent_beta:g}",
            "repair_gamma": f"{gamma:g}",
            "site_gammas_sha256": sha256_json(overrides),
            "parent_factors_sha256": sha256_file(parent.FACTORS),
            "repair_detectors_sha256": sha256_file(REPAIR_DETECTORS),
        },
    )
    plan = GGUFQuantizedAblationPlan(
        source_sha256=str(preparation["base_q8"]["sha256"]),
        tensor_artifact_sha256=sha256_file(factors),
        edits=tuple(edits),
        row_chunk_size=256,
        verify_untouched_bytes=not fast_search,
    )
    plan_path = RUN_DIR / f"{tag}.plan.json"
    plan.write(plan_path)
    output = ROOT / "outputs" / f"gemma4-e2b-q8-{tag}.gguf"
    merge = apply_quantized_gguf_ablation(
        parent.BASE_Q8,
        output,
        plan_path,
        factors,
        fast_search=fast_search,
    )
    report = {
        "schema_version": "gemma4-e2b-q8-additive-repair-build-v1",
        "model": {"id": parent.MODEL_ID, "revision": parent.MODEL_REVISION},
        "base_q8": preparation["base_q8"],
        "parent_factors": {
            "path": str(parent.FACTORS),
            "sha256": sha256_file(parent.FACTORS),
        },
        "repair_detectors": {
            "path": str(REPAIR_DETECTORS),
            "sha256": sha256_file(REPAIR_DETECTORS),
            "diagnostics": str(REPAIR_DIAGNOSTICS),
            "diagnostics_sha256": sha256_file(REPAIR_DIAGNOSTICS),
        },
        "composition": composition,
        "fast_search": fast_search,
        "factor_artifact": {"path": str(factors), "sha256": sha256_file(factors)},
        "merge": merge,
    }
    report_path = RUN_DIR / f"{tag}.build.json"
    write_json(report_path, report)
    return {"tag": tag, "output": str(output), "report": str(report_path), **merge}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-beta", type=float, default=DEFAULT_PARENT_BETA)
    parser.add_argument("--gammas", required=True)
    parser.add_argument("--site-gamma", action="append", default=[])
    parser.add_argument("--fast-search", action="store_true")
    parser.add_argument("--tag")
    args = parser.parse_args()
    site_gammas: dict[str, float] = {}
    for raw in args.site_gamma:
        site_id, separator, value = raw.partition("=")
        if not separator or site_id in site_gammas:
            raise ValueError(f"invalid or duplicate --site-gamma: {raw}")
        site_gammas[site_id] = float(value)
    if args.tag and len(args.gammas.split(",")) != 1:
        raise ValueError("--tag requires exactly one gamma")
    for raw in args.gammas.split(","):
        result = build(
            args.parent_beta,
            float(raw),
            site_gammas=site_gammas,
            fast_search=args.fast_search,
            tag=args.tag,
        )
        print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
