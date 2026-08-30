#!/usr/bin/env python3
"""Fit the original PRIME consensus/metric operators for direct Q8 editing."""

from __future__ import annotations

import json
import math
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import load_file, save_file
import torch

from experiments.lfm25_2p6b_residual_stream import (
    BAD_DATASET,
    BAD_REVISION,
    GOOD_DATASET,
    GOOD_REVISION,
)
from heretic_nx.data.research_splits import ResearchSplitManifest
from heretic_nx.edits.activation_op import metric_projector_operator
from heretic_nx.geometry.consensus import grassmann_consensus
from heretic_nx.geometry.metric import (
    LowRankMetric,
    MetricGeometryGate,
    require_static_geometry,
)
from heretic_nx.geometry.pca import exact_principal_components
from heretic_nx.hashing import (
    canonical_json,
    sha256_directory,
    sha256_file,
    sha256_json,
)


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "runs" / "lfm25-8b-a1b-q8-direct"
PROGRESS = RUN_DIR / "mlx-site-activations.progress.json"
ACTIVATIONS = RUN_DIR / "mlx-site-activations.safetensors"
OUTPUT = RUN_DIR / "prime-site-operators.safetensors"
REPORT = RUN_DIR / "prime-site-operators.json"
FOLDS = 3
CAPABILITY_RANK = 8
REGULARIZATION = 1e-3
_RUNTIME_PROGRESS_KEYS = frozenset(
    {"label", "completed", "seconds", "complete", "output_sha256"}
)


def load_activation_provenance() -> tuple[dict[str, object], dict[str, object]]:
    """Validate the complete v2 activation cache before fitting operators."""

    progress = json.loads(PROGRESS.read_text(encoding="utf-8"))
    if progress.get("schema_version") != "lfm25-8b-a1b-mlx-sites-progress-v2":
        raise RuntimeError("semantic activations predate leak-safe geometry splits")
    if (
        progress.get("complete") is not True
        or progress.get("label") != "target"
        or progress.get("completed") != progress.get("count")
    ):
        raise RuntimeError("semantic activation collection is incomplete")
    immutable = {
        key: value
        for key, value in progress.items()
        if key not in _RUNTIME_PROGRESS_KEYS
    }
    immutable_sha256 = sha256_json(immutable)
    if progress.get("output_sha256") != sha256_file(ACTIVATIONS):
        raise RuntimeError("semantic activation artifact hash mismatch")

    safe_manifest = ResearchSplitManifest.from_dict(
        progress["safe_split_manifest"]
    )
    target_manifest = ResearchSplitManifest.from_dict(
        progress["target_split_manifest"]
    )
    count = progress.get("count")
    pool_count = progress.get("pool_count")
    split_seed = progress.get("split_seed")
    if (
        safe_manifest.sha256 != progress.get("safe_split_manifest_sha256")
        or target_manifest.sha256
        != progress.get("target_split_manifest_sha256")
        or safe_manifest.purpose != "geometry"
        or target_manifest.purpose != "geometry"
        or safe_manifest.dataset_id != GOOD_DATASET
        or safe_manifest.revision != GOOD_REVISION
        or target_manifest.dataset_id != BAD_DATASET
        or target_manifest.revision != BAD_REVISION
        or safe_manifest.source_split != "train"
        or target_manifest.source_split != "train"
        or safe_manifest.seed != split_seed
        or target_manifest.seed != split_seed
        or safe_manifest.pool_size != pool_count
        or target_manifest.pool_size != pool_count
        or len(safe_manifest.rows) != count
        or len(target_manifest.rows) != count
    ):
        raise RuntimeError("semantic activation split provenance is inconsistent")

    model = Path(str(progress["model"]))
    source = Path(str(progress["source"]))
    if (
        not model.is_dir()
        or progress.get("model_sha256") != sha256_directory(model)
        or not source.is_file()
        or progress.get("source_sha256") != sha256_file(source)
    ):
        raise RuntimeError("semantic activation model provenance is stale")

    with safe_open(ACTIVATIONS, framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
    required_metadata = {
        "manifest_sha256": immutable_sha256,
        "sites_sha256": str(progress["sites_sha256"]),
        "safe_split_manifest_sha256": safe_manifest.sha256,
        "target_split_manifest_sha256": target_manifest.sha256,
        "model_sha256": str(progress["model_sha256"]),
        "source_sha256": str(progress["source_sha256"]),
    }
    if any(metadata.get(key) != value for key, value in required_metadata.items()):
        raise RuntimeError("semantic activation tensors and manifest disagree")
    if progress.get("sites_sha256") != sha256_json(progress.get("sites")):
        raise RuntimeError("semantic site manifest hash mismatch")
    return progress, {
        "activation_manifest_sha256": immutable_sha256,
        "safe_split_manifest": safe_manifest.to_dict(),
        "safe_split_manifest_sha256": safe_manifest.sha256,
        "target_split_manifest": target_manifest.to_dict(),
        "target_split_manifest_sha256": target_manifest.sha256,
        "model": str(model.resolve()),
        "model_sha256": str(progress["model_sha256"]),
        "source": str(source.resolve()),
        "source_sha256": str(progress["source_sha256"]),
        "sites_sha256": str(progress["sites_sha256"]),
    }


def main() -> None:
    manifest, provenance = load_activation_provenance()
    specs = list(manifest["sites"])
    cached = load_file(ACTIVATIONS)
    safe = cached["safe"].float()
    target = cached["target"].float()
    if safe.shape != target.shape or safe.shape[1] != len(specs):
        raise RuntimeError("semantic activation cache and site manifest disagree")
    if safe.shape[0] != manifest["count"] or safe.shape[2] != manifest["width"]:
        raise RuntimeError("semantic activation cache has an invalid shape")

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

        capability_fit = exact_principal_components(
            safe_values,
            maximum_rank=CAPABILITY_RANK,
        )
        capability = capability_fit.basis
        singular = capability_fit.singular_values
        if capability_fit.effective_rank == 0:
            rejected.append({**spec, "reason": "empty-capability-subspace"})
            continue
        covariance_factor = capability * (
            singular / math.sqrt(max(safe_values.shape[0] - 1, 1))
        )[None, :]
        metric = LowRankMetric.from_factors(
            safe_values.shape[1],
            covariance_factor=covariance_factor,
            regularization=REGULARIZATION,
        )
        geometry = gate.evaluate(consensus.basis, capability, metric)
        try:
            editable = require_static_geometry(
                geometry,
                site_id=str(spec["site_id"]),
            )
        except RuntimeError:
            rejected.append(
                {
                    **spec,
                    "reason": "metric-geometry-gate",
                    "current_gate_decision": geometry.decision,
                    "minimum_angle_deg": geometry.minimum_angle_deg,
                    "retained_energy": geometry.retained_energy,
                    "capability_effective_rank": capability_fit.effective_rank,
                    "capability_retained_energy_fraction": (
                        capability_fit.retained_energy_fraction
                    ),
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
                "selection_gate": "safe-static-metric-geometry",
                "capability_effective_rank": capability_fit.effective_rank,
                "capability_retained_energy_fraction": (
                    capability_fit.retained_energy_fraction
                ),
                "consensus_mass": consensus.captured_stability_mass,
            }
        )
    accepted.sort(key=lambda row: (-float(row["score"]), str(row["site_id"])))
    save_file(
        tensors,
        OUTPUT,
        metadata={
            "activations_sha256": sha256_file(ACTIVATIONS),
            "activation_manifest_sha256": str(
                provenance["activation_manifest_sha256"]
            ),
            "safe_split_manifest_sha256": str(
                provenance["safe_split_manifest_sha256"]
            ),
            "target_split_manifest_sha256": str(
                provenance["target_split_manifest_sha256"]
            ),
            "source_sha256": str(provenance["source_sha256"]),
            "sites_sha256": str(provenance["sites_sha256"]),
        },
    )
    report = {
        "schema_version": "lfm25-8b-a1b-prime-sites-v2",
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
            "manifest_sha256": provenance["activation_manifest_sha256"],
        },
        "geometry_provenance": provenance,
        "accepted": accepted,
        "rejected": rejected,
        "operator_artifact": {
            "path": str(OUTPUT),
            "sha256": sha256_file(OUTPUT),
        },
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
