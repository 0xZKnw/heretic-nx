#!/usr/bin/env python3
"""Build a benign-penalized direct-Q8 teacher-delta candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.lfm25_8b_a1b_distill_teacher import OUTPUT as FACTORS
from experiments.lfm25_8b_a1b_distill_teacher import REPORT as DISTILL_REPORT
from experiments.lfm25_8b_a1b_q8_build import RUN_DIR, SOURCE
from experiments.lfm25_8b_a1b_teacher_inputs import TEACHER_MERGE
from heretic_nx.edits import GGUFQ8AblationPlan, GGUFQ8TensorEdit, apply_q8_gguf_ablation
from heretic_nx.hashing import canonical_json, sha256_file


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
    required = (SOURCE, FACTORS, DISTILL_REPORT, TEACHER_MERGE)
    if not all(path.is_file() for path in required):
        raise RuntimeError("distilled teacher factors are missing")
    distill = json.loads(DISTILL_REPORT.read_text(encoding="utf-8"))
    if args.safe_lambda not in [float(value) for value in distill["safe_lambdas"]]:
        raise ValueError("safe-lambda was not fitted")
    teacher = json.loads(TEACHER_MERGE.read_text(encoding="utf-8"))
    selected = list(teacher["candidate"]["selected"])[: args.k]
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
        "teacher_merge": str(TEACHER_MERGE),
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
