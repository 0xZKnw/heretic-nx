#!/usr/bin/env python3
"""Build one site-ranked PRIME candidate directly in the target Q8 GGUF."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gguf import GGMLQuantizationType, GGUFReader

from experiments.lfm25_8b_a1b_prime_sites import OUTPUT as OPERATORS
from experiments.lfm25_8b_a1b_prime_sites import REPORT as RANKING_REPORT
from experiments.lfm25_8b_a1b_q8_build import ROOT, RUN_DIR, SOURCE
from heretic_nx.edits import (
    GGUFQ8AblationPlan,
    GGUFQ8TensorEdit,
    apply_q8_gguf_ablation,
)
from heretic_nx.hashing import canonical_json, sha256_file


def tensor_name(row: dict[str, object]) -> str:
    layer = int(row["layer"])
    family = str(row["family"])
    if family == "gqa":
        return f"blk.{layer}.attn_output.weight"
    if family == "liv":
        return f"blk.{layer}.shortconv.out_proj.weight"
    if family == "ffn":
        suffix = "ffn_down.weight" if layer < 2 else "ffn_down_exps.weight"
        return f"blk.{layer}.{suffix}"
    raise ValueError(f"unsupported PRIME family: {family}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--beta", type=float, required=True)
    parser.add_argument(
        "--max-safe-drift",
        type=float,
        help="discard PRIME sites whose calibration safe-drift exceeds this value",
    )
    parser.add_argument(
        "--families",
        help="comma-separated PRIME families to retain (ffn,gqa,liv)",
    )
    parser.add_argument(
        "--exclude-sites",
        help="comma-separated semantic site ids to discard",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.k <= 0 or args.beta < 0:
        raise ValueError("k must be positive and beta must be non-negative")
    if args.max_safe_drift is not None and args.max_safe_drift <= 0:
        raise ValueError("max-safe-drift must be positive")
    families = (
        {value.strip() for value in args.families.split(",") if value.strip()}
        if args.families
        else None
    )
    if families is not None and not families <= {"ffn", "gqa", "liv"}:
        raise ValueError("families must contain only ffn, gqa, or liv")
    excluded = (
        {value.strip() for value in args.exclude_sites.split(",") if value.strip()}
        if args.exclude_sites
        else set()
    )

    ranking = json.loads(RANKING_REPORT.read_text(encoding="utf-8"))["accepted"]
    reader = GGUFReader(SOURCE)
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    editable = []
    filtered = []
    skipped = []
    for row in ranking:
        if (
            args.max_safe_drift is not None
            and float(row["safe_proxy_drift"]) > args.max_safe_drift
        ):
            filtered.append({"site_id": row["site_id"], "reason": "safe-drift"})
            continue
        if families is not None and str(row["family"]) not in families:
            filtered.append({"site_id": row["site_id"], "reason": "family"})
            continue
        if str(row["site_id"]) in excluded:
            filtered.append({"site_id": row["site_id"], "reason": "explicit-exclusion"})
            continue
        name = tensor_name(row)
        tensor = tensors.get(name)
        if tensor is None or tensor.tensor_type != GGMLQuantizationType.Q8_0:
            skipped.append(
                {
                    "site_id": row["site_id"],
                    "tensor_name": name,
                    "reason": "missing" if tensor is None else tensor.tensor_type.name,
                }
            )
            continue
        editable.append({**row, "tensor_name": name})
    selected = editable[: min(args.k, len(editable))]
    if not selected:
        raise RuntimeError("the PRIME ranking exposes no editable Q8 sites")
    edits = tuple(
        GGUFQ8TensorEdit(
            tensor_name=str(row["tensor_name"]),
            a_key=str(row["factor_a_key"]),
            b_key=str(row["factor_b_key"]),
            strength=args.beta,
            preserve_row_norms=False,
        )
        for row in selected
    )
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    plan_path = RUN_DIR / f"{args.label}.plan.json"
    report_path = RUN_DIR / f"{args.label}.merge.json"
    plan = GGUFQ8AblationPlan(
        source_sha256=sha256_file(SOURCE),
        tensor_artifact_sha256=sha256_file(OPERATORS),
        edits=edits,
    )
    plan.write(plan_path)
    report = apply_q8_gguf_ablation(
        SOURCE,
        args.output,
        plan_path,
        OPERATORS,
        force=args.force,
    )
    report["candidate"] = {
        "label": args.label,
        "k": args.k,
        "beta": args.beta,
        "active_sites": len(selected),
        "preserve_row_norms": False,
        "max_safe_drift": args.max_safe_drift,
        "families": sorted(families) if families is not None else None,
        "excluded_sites": sorted(excluded),
        "selected": selected,
        "filtered": filtered,
        "skipped_non_q8": skipped,
        "ranking_report": str(RANKING_REPORT),
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
