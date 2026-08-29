#!/usr/bin/env python3
"""Fit and rank per-site dense axes for Ling-3.0-tiny Q8 research."""

from __future__ import annotations

import json
from pathlib import Path

from safetensors.torch import load_file, save_file

from experiments.ling3_tiny_native_dense_sites import OUTPUT as SITE_OUTPUTS
from experiments.ling3_tiny_native_dense_sites import REPORT as SITE_REPORT
from experiments.ling3_tiny_native_dense_sites import RUN_DIR
from heretic_nx.geometry.residual import (
    fit_residual_stream_axes,
    protect_residual_stream_axes,
)
from heretic_nx.hashing import canonical_json, sha256_file


OUTPUT = RUN_DIR / "native-dense-site-axes.safetensors"
REPORT = RUN_DIR / "native-dense-site-axes.json"
RANKS = (1, 4, 8)


def main() -> None:
    if not SITE_OUTPUTS.is_file() or not SITE_REPORT.is_file():
        raise RuntimeError("native dense site outputs are missing")
    manifest = json.loads(SITE_REPORT.read_text(encoding="utf-8"))
    values = load_file(SITE_OUTPUTS)
    safe = values["safe"].float()
    target = values["target"].float()
    if safe.shape != target.shape or safe.ndim != 3 or safe.shape[1:] != (48, 1536):
        raise RuntimeError(f"unexpected native site outputs: {tuple(safe.shape)}")

    raw_axes = fit_residual_stream_axes(
        safe,
        target,
        folds=3,
        remove_safe_mean=True,
    )
    protected_by_rank = {
        rank: protect_residual_stream_axes(
            safe,
            target,
            raw_axes,
            capability_rank=rank,
            seed=7300 + 100 * rank,
            device="cpu",
        )
        for rank in RANKS
    }
    payload = {}
    rows = []
    for index, (site_id, tensor_name, raw) in enumerate(
        zip(manifest["site_ids"], manifest["tensor_names"], raw_axes)
    ):
        key = f"site{index:02d}"
        payload[f"{key}.raw"] = raw.axis.float().contiguous()
        diagnostics = {
            "index": index,
            "site_id": site_id,
            "tensor_name": tensor_name,
            "family": "attention" if site_id.endswith("attention_out") else "dense_ffn",
            "layer": index // 2,
            "raw_fold_cosine_minimum": raw.fold_cosine_minimum,
            "raw_fold_cosine_mean": raw.fold_cosine_mean,
            "raw_safe_mean_cosine": raw.safe_mean_cosine,
            "protected": {},
        }
        for rank in RANKS:
            protected = protected_by_rank[rank][index]
            payload[f"{key}.r{rank}"] = protected.evidence.axis.float().contiguous()
            diagnostics["protected"][f"r{rank}"] = {
                "retained_fraction": protected.retained_fraction,
                "safe_projection_rms": protected.safe_projection_rms,
                "target_separation": protected.target_separation,
                "efficiency": protected.efficiency,
                "safe_mean_cosine": protected.evidence.safe_mean_cosine,
            }
        rows.append(diagnostics)

    for rank in RANKS:
        field = f"r{rank}"
        ordering = sorted(
            range(len(rows)),
            key=lambda index: (
                rows[index]["protected"][field]["efficiency"]
                * max(rows[index]["raw_fold_cosine_minimum"], 0.0)
            ),
            reverse=True,
        )
        for order, index in enumerate(ordering, start=1):
            rows[index]["protected"][field]["score_rank"] = order
            rows[index]["protected"][field]["score"] = (
                rows[index]["protected"][field]["efficiency"]
                * max(rows[index]["raw_fold_cosine_minimum"], 0.0)
            )

    save_file(
        payload,
        OUTPUT,
        metadata={"site_outputs_sha256": sha256_file(SITE_OUTPUTS)},
    )
    report = {
        "schema_version": "ling3-tiny-native-dense-site-axes-v1",
        "research_only_native_source": str(SITE_OUTPUTS),
        "research_only_native_source_sha256": sha256_file(SITE_OUTPUTS),
        "candidate_weight_domain": "pinned Q8 GGUF only",
        "rows": rows,
        "top": {
            f"r{rank}": [
                {
                    "site_id": row["site_id"],
                    "tensor_name": row["tensor_name"],
                    **row["protected"][f"r{rank}"],
                }
                for row in sorted(
                    rows,
                    key=lambda row: row["protected"][f"r{rank}"]["score_rank"],
                )[:16]
            ]
            for rank in RANKS
        },
        "output": str(OUTPUT.resolve()),
    }
    REPORT.write_bytes(canonical_json(report) + b"\n")
    print(json.dumps(report["top"], indent=2), flush=True)


if __name__ == "__main__":
    main()
