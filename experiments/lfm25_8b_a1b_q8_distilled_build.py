#!/usr/bin/env python3
"""Build a benign-penalized direct-Q8 teacher-delta candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from safetensors import safe_open

from experiments.lfm25_8b_a1b_distill_teacher import OUTPUT as FACTORS
from experiments.lfm25_8b_a1b_distill_teacher import REPORT as DISTILL_REPORT
from experiments.lfm25_8b_a1b_distill_teacher import INPUTS, INPUT_MANIFEST, OPERATORS
from experiments.lfm25_8b_a1b_q8_build import RUN_DIR, SOURCE
from experiments.lfm25_8b_a1b_q8_prime_build import (
    RANKING_REPORT,
    load_verified_prime_merge,
)
from experiments.lfm25_8b_a1b_teacher_inputs import TEACHER_MERGE
from heretic_nx.edits import GGUFQ8AblationPlan, GGUFQ8TensorEdit, apply_q8_gguf_ablation
from heretic_nx.hashing import canonical_json, sha256_file, sha256_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--safe-lambda", type=float, required=True)
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.safe_lambda <= 0 or args.beta < 0 or not 1 <= args.k <= 8:
        raise ValueError("safe-lambda must be positive, beta non-negative, and k in 1..8")
    required = (
        SOURCE,
        FACTORS,
        DISTILL_REPORT,
        TEACHER_MERGE,
        INPUTS,
        INPUT_MANIFEST,
        OPERATORS,
        RANKING_REPORT,
    )
    if not all(path.is_file() for path in required):
        raise RuntimeError("distilled teacher factors are missing")
    distill = json.loads(DISTILL_REPORT.read_text(encoding="utf-8"))
    teacher, ranking = load_verified_prime_merge(
        TEACHER_MERGE,
        verify_output=True,
    )
    selected_all = list(teacher["candidate"]["selected"])
    selected_site_ids = [str(row["site_id"]) for row in selected_all]
    selected_sites_sha256 = sha256_json(selected_all)
    input_manifest = json.loads(INPUT_MANIFEST.read_text(encoding="utf-8"))
    input_output_sha256 = input_manifest.pop("output_sha256", None)
    input_manifest_sha256 = input_manifest.pop("manifest_sha256", None)
    if (
        input_manifest.get("schema_version")
        != "lfm25-8b-a1b-teacher-op-inputs-v3"
        or input_manifest_sha256 != sha256_json(input_manifest)
        or input_output_sha256 != sha256_file(INPUTS)
    ):
        raise RuntimeError("teacher input manifest or tensor hash is stale")
    factor_sha256 = sha256_file(FACTORS)
    teacher_merge_sha256 = sha256_file(TEACHER_MERGE)
    ranking_sha256 = sha256_file(RANKING_REPORT)
    operators_sha256 = sha256_file(OPERATORS)
    source_sha256 = str(teacher["source"]["sha256"])
    if (
        distill.get("schema_version")
        != "lfm25-8b-a1b-teacher-op-distillation-v3"
        or distill.get("inputs", {}).get("manifest_sha256")
        != input_manifest_sha256
        or distill.get("inputs", {}).get("sha256") != input_output_sha256
        or Path(str(distill.get("inputs", {}).get("path", ""))).resolve()
        != INPUTS.resolve()
        or distill.get("output", {}).get("sha256") != factor_sha256
        or Path(str(distill.get("output", {}).get("path", ""))).resolve()
        != FACTORS.resolve()
        or distill.get("teacher", {}).get("merge_sha256")
        != teacher_merge_sha256
        or distill.get("teacher", {}).get("source_artifact_sha256")
        != source_sha256
        or distill.get("teacher", {}).get("artifact_sha256")
        != teacher["output"]["sha256"]
        or distill.get("teacher", {}).get("operators_sha256")
        != operators_sha256
        or distill.get("teacher", {}).get("ranking_report_sha256")
        != ranking_sha256
        or distill.get("teacher", {}).get("geometry_provenance_sha256")
        != sha256_json(ranking["geometry_provenance"])
        or distill.get("teacher", {}).get("selected_sites_sha256")
        != selected_sites_sha256
        or distill.get("teacher", {}).get("sites") != selected_site_ids
    ):
        raise RuntimeError(
            "distilled factors are not bound to the leak-safe input protocol"
        )
    if args.safe_lambda not in [float(value) for value in distill["safe_lambdas"]]:
        raise ValueError("safe-lambda was not fitted")
    expected_factor_keys = {
        f"site{index:02d}.axis" for index in range(len(selected_all))
    } | {
        f"lambda{safe_lambda:g}.site{index:02d}.right"
        for safe_lambda in (float(value) for value in distill["safe_lambdas"])
        for index in range(len(selected_all))
    }
    with safe_open(FACTORS, framework="pt", device="cpu") as handle:
        factor_metadata = handle.metadata() or {}
        if set(handle.keys()) != expected_factor_keys:
            raise RuntimeError("distilled factor keys do not match the teacher sites")
        for key in expected_factor_keys:
            if tuple(handle.get_tensor(key).shape) != (2048, 1):
                raise RuntimeError(f"invalid distilled factor shape: {key}")
    expected_factor_metadata = {
        "inputs_sha256": input_output_sha256,
        "inputs_manifest_sha256": input_manifest_sha256,
        "operators_sha256": operators_sha256,
        "teacher_merge_sha256": teacher_merge_sha256,
        "ranking_report_sha256": ranking_sha256,
        "source_artifact_sha256": source_sha256,
        "teacher_artifact_sha256": str(teacher["output"]["sha256"]),
        "selected_sites_sha256": selected_sites_sha256,
    }
    if any(
        factor_metadata.get(key) != value
        for key, value in expected_factor_metadata.items()
    ):
        raise RuntimeError("distilled factors and provenance metadata disagree")
    selected = selected_all[: args.k]
    edits = tuple(
        GGUFQ8TensorEdit(
            tensor_name=str(row["tensor_name"]),
            a_key=f"site{index:02d}.axis",
            right_key=f"lambda{args.safe_lambda:g}.site{index:02d}.right",
            strength=args.beta,
            preserve_row_norms=False,
        )
        for index, row in enumerate(selected)
    )
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    plan_path = RUN_DIR / f"{args.label}.plan.json"
    report_path = RUN_DIR / f"{args.label}.merge.json"
    plan = GGUFQ8AblationPlan(
        source_sha256=sha256_file(SOURCE),
        tensor_artifact_sha256=sha256_file(FACTORS),
        edits=edits,
    )
    plan.write(plan_path)
    report = apply_q8_gguf_ablation(
        SOURCE,
        args.output,
        plan_path,
        FACTORS,
        force=args.force,
    )
    report["candidate"] = {
        "label": args.label,
        "safe_lambda": args.safe_lambda,
        "beta": args.beta,
        "k": args.k,
        "active_sites": len(selected),
        "selected": selected,
        "distillation_report": str(DISTILL_REPORT),
        "distillation_report_sha256": sha256_file(DISTILL_REPORT),
        "distilled_factors_sha256": factor_sha256,
        "input_manifest_sha256": input_manifest_sha256,
        "teacher_merge": str(TEACHER_MERGE),
        "teacher_merge_sha256": teacher_merge_sha256,
        "ranking_report_sha256": ranking_sha256,
        "operator_artifact_sha256": operators_sha256,
        "selected_sites_sha256": sha256_json(selected),
    }
    report_path.write_bytes(canonical_json(report) + b"\n")
    print(
        json.dumps(
            {
                "candidate": report["candidate"],
                "output": report["output"],
                "report": str(report_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
