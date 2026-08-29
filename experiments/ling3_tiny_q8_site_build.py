#!/usr/bin/env python3
"""Build sparse per-site Ling-3.0-tiny candidates directly in Q8."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from heretic_nx.edits import (
    GGUFQ8AblationPlan,
    GGUFQ8TensorEdit,
    apply_q8_gguf_ablation,
)
from heretic_nx.hashing import canonical_json, sha256_file

from experiments.ling3_tiny_native_dense_axes import OUTPUT as FACTORS
from experiments.ling3_tiny_native_dense_axes import REPORT as AXIS_REPORT
from experiments.ling3_tiny_q8_build import RUN_DIR, SOURCE


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--rank", choices=("r1", "r4", "r8"), default="r8")
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--beta", type=float, required=True)
    parser.add_argument("--ffn-multiplier", type=float, default=1.0)
    parser.add_argument("--preserve-row-norms", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.k <= 48 or args.beta < 0 or args.ffn_multiplier < 0:
        raise ValueError("k must be in 1..48 and strengths must be non-negative")
    required = (SOURCE, FACTORS, AXIS_REPORT)
    if not all(path.is_file() for path in required):
        raise RuntimeError("Q8 source or native dense site axes are missing")

    report = json.loads(AXIS_REPORT.read_text(encoding="utf-8"))
    field = args.rank
    selected = sorted(
        report["rows"],
        key=lambda row: row["protected"][field]["score_rank"],
    )[: args.k]
    edits = tuple(
        GGUFQ8TensorEdit(
            tensor_name=str(row["tensor_name"]),
            a_key=f"site{int(row['index']):02d}.{field}",
            strength=args.beta
            * (args.ffn_multiplier if row["family"] == "dense_ffn" else 1.0),
            preserve_row_norms=args.preserve_row_norms,
        )
        for row in selected
    )
    plan_path = RUN_DIR / f"{args.label}.plan.json"
    merge_path = RUN_DIR / f"{args.label}.merge.json"
    plan = GGUFQ8AblationPlan(
        source_sha256=sha256_file(SOURCE),
        tensor_artifact_sha256=sha256_file(FACTORS),
        edits=edits,
    )
    plan.write(plan_path)
    merge = apply_q8_gguf_ablation(
        SOURCE,
        args.output,
        plan_path,
        FACTORS,
        force=args.force,
    )
    merge["candidate"] = {
        "label": args.label,
        "rank": args.rank,
        "k": args.k,
        "beta": args.beta,
        "ffn_multiplier": args.ffn_multiplier,
        "preserve_row_norms": args.preserve_row_norms,
        "routed_experts_edited": False,
        "selected": [
            {
                "site_id": row["site_id"],
                "tensor_name": row["tensor_name"],
                "score": row["protected"][field]["score"],
                "strength": args.beta
                * (args.ffn_multiplier if row["family"] == "dense_ffn" else 1.0),
            }
            for row in selected
        ],
    }
    merge_path.write_bytes(canonical_json(merge) + b"\n")
    print(
        json.dumps(
            {
                "candidate": merge["candidate"],
                "output": merge["output"],
                "report": str(merge_path),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
