#!/usr/bin/env python3
"""Fit the original PRIME consensus/metric operators for direct Q8 editing."""

from __future__ import annotations

import json
import math
from pathlib import Path

from safetensors.torch import load_file, save_file
import torch

from experiments.lfm25_8b_a1b_mlx_sites import PROGRESS, RUN_DIR
from heretic_nx.edits.activation_op import metric_projector_operator
from heretic_nx.geometry.consensus import grassmann_consensus
from heretic_nx.geometry.metric import (
    LowRankMetric,
    MetricGeometryGate,
    metric_residualize,
)
from heretic_nx.hashing import canonical_json, sha256_file


ACTIVATIONS = RUN_DIR / "mlx-site-activations.safetensors"
OUTPUT = RUN_DIR / "prime-site-operators.safetensors"
REPORT = RUN_DIR / "prime-site-operators.json"
FOLDS = 3
CAPABILITY_RANK = 8
REGULARIZATION = 1e-3


def main() -> None:
    manifest = json.loads(PROGRESS.read_text(encoding="utf-8"))
    specs = list(manifest["sites"])
    cached = load_file(ACTIVATIONS)
    safe = cached["safe"].float()
    target = cached["target"].float()
    if safe.shape != target.shape or safe.shape[1] != len(specs):
        raise RuntimeError("semantic activation cache and site manifest disagree")

    gate = MetricGeometryGate()
    tensors = {}
    accepted = []
    rejected = []
    for site_index, spec in enumerate(specs):
        safe_values = safe[:, site_index]
        target_values = target[:, site_index]
        fold_bases = []
        for fold in range(FOLDS):
            direction = (
                target_values[fold::FOLDS].mean(dim=0)
                - safe_values[fold::FOLDS].mean(dim=0)
            )
            norm = torch.linalg.vector_norm(direction)
            if float(norm) > 1e-7:
                fold_bases.append(direction[:, None] / norm)
        if len(fold_bases) != FOLDS:
            rejected.append({**spec, "reason": "collapsed-fold"})
            continue
        consensus = grassmann_consensus(
            fold_bases,
            eigenvalue_minimum=0.25,
            stability_mass=0.80,
            maximum_rank=2,
        )
        if consensus.selected_rank == 0:
            rejected.append({**spec, "reason": "empty-consensus"})
            continue

        centered = safe_values - safe_values.mean(dim=0)
        torch.manual_seed(8258 + site_index)
        _u, singular, vectors = torch.pca_lowrank(
            centered,
            q=min(CAPABILITY_RANK + 4, centered.shape[0], centered.shape[1]),
            center=False,
            niter=3,
        )
        capability = vectors[:, :CAPABILITY_RANK]
        singular = singular[:CAPABILITY_RANK]
        covariance_factor = capability * (
            singular / math.sqrt(max(safe_values.shape[0] - 1, 1))
        )[None, :]
        metric = LowRankMetric.from_factors(
            safe_values.shape[1],
            covariance_factor=covariance_factor,
            regularization=REGULARIZATION,
        )
        geometry = gate.evaluate(consensus.basis, capability, metric)
        editable = metric_residualize(consensus.basis, capability, metric)
        if editable.shape[1] == 0:
            rejected.append(
                {
                    **spec,
                    "reason": "collapsed-metric-residual",
                    "current_gate_decision": geometry.decision,
                    "minimum_angle_deg": geometry.minimum_angle_deg,
                    "retained_energy": geometry.retained_energy,
                }
            )
            continue
        operator = metric_projector_operator(editable, metric, beta=1.0)
        a = operator.a.float().cpu().contiguous()
        b = operator.b.float().cpu().contiguous()
        safe_delta = (safe_values @ b) @ a.T
        target_delta = (target_values @ b) @ a.T
        safe_drift = float(
            torch.linalg.vector_norm(safe_delta)
            / torch.linalg.vector_norm(safe_values).clamp_min(1e-8)
        )
        target_drift = float(
            torch.linalg.vector_norm(target_delta)
            / torch.linalg.vector_norm(target_values).clamp_min(1e-8)
        )
        separation = target_values.mean(dim=0) - safe_values.mean(dim=0)
        target_effect = float(torch.linalg.vector_norm(separation @ b))
        score = target_effect * max(target_drift, 1e-8) / max(safe_drift, 1e-6)
        key = f"site{site_index:02d}"
        tensors[f"{key}.a"] = a
        tensors[f"{key}.b"] = b
        accepted.append(
            {
                **spec,
                "factor_a_key": f"{key}.a",
                "factor_b_key": f"{key}.b",
                "rank": int(a.shape[1]),
                "score": score,
                "safe_proxy_drift": safe_drift,
                "target_proxy_drift": target_drift,
                "target_effect": target_effect,
                "minimum_angle_deg": geometry.minimum_angle_deg,
                "retained_energy": geometry.retained_energy,
                "current_gate_decision": geometry.decision,
                "selection_gate": "legacy-prime-metric-residual",
                "consensus_mass": consensus.captured_stability_mass,
            }
        )
    accepted.sort(key=lambda row: (-float(row["score"]), str(row["site_id"])))
    save_file(
        tensors,
        OUTPUT,
        metadata={"activations_sha256": sha256_file(ACTIVATIONS)},
    )
    report = {
        "schema_version": "lfm25-8b-a1b-prime-sites-v1",
        "method": {
            "folds": FOLDS,
            "capability_rank": CAPABILITY_RANK,
            "metric_regularization": REGULARIZATION,
            "row_norm_preservation": False,
        },
        "activations": {
            "path": str(ACTIVATIONS),
            "sha256": sha256_file(ACTIVATIONS),
            "shape": list(safe.shape),
        },
        "accepted": accepted,
        "rejected": rejected,
        "operator_artifact": str(OUTPUT),
    }
    REPORT.write_bytes(canonical_json(report) + b"\n")
    print(
        json.dumps(
            {
                "accepted": len(accepted),
                "rejected": len(rejected),
                "top_sites": accepted[:16],
                "operators": str(OUTPUT),
                "report": str(REPORT),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
