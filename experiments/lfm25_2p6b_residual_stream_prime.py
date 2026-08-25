#!/usr/bin/env python3
"""Sparse capability-protected Residual-Stream search for LFM2.5-2.6B.

Residual-Stream remains the intervention core. PRIME protects the dominant
benign subspace and sparsely routes edits to semantic sites with the highest
target-effect-to-benign-drift ratio under a hard first-token KL budget.
"""

from __future__ import annotations

from dataclasses import replace
import gc
import json
import math
from pathlib import Path
import shutil
import time
from typing import Any

from datasets import load_dataset
from safetensors.torch import load_file, save_file
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.lfm25_2p6b_residual_stream import (
    BAD_DATASET,
    BAD_REVISION,
    COLLECT_COUNT,
    FOLD_A,
    FOLD_B,
    FOLDS,
    GOOD_DATASET,
    GOOD_REVISION,
    HOLDOUT,
    MODEL_ID,
    MODEL_REVISION,
    REFERENCE_KERNELS,
    REFERENCE_LAYER_COUNT,
    base_path,
    build_editors,
    collect_residuals,
    mean_first_token_kl,
    next_token_log_probs,
    refusal_evaluation,
)
from heretic_nx.edits.activation_op import metric_projector_operator
from heretic_nx.edits.residual_stream import (
    apply_residual_stream_weight_edits,
    snapshot_residual_stream_weights,
)
from heretic_nx.geometry.contrastive import ContrastiveAxis
from heretic_nx.geometry.metric import (
    LowRankMetric,
    metric_residualize,
)
from heretic_nx.geometry.residual import (
    fit_residual_stream_axes,
    protect_residual_stream_axes,
)
from heretic_nx.hashing import canonical_json, sha256_file, sha256_json


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
PARENT_RUN_DIR = ROOT / "runs" / "lfm25-2p6b-residual-stream"
PARENT_REPORT = PARENT_RUN_DIR / "report.json"
RUN_DIR = ROOT / "runs" / "lfm25-2p6b-residual-stream-prime-v4"
REPORT = RUN_DIR / "report.json"
RESIDUAL_CACHE = (
    ROOT
    / "runs"
    / "lfm25-2p6b-residual-stream-prime-v2"
    / "residuals-closed-think.safetensors"
)
OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "LFM2.5-2.6B-Heretic-NX-Residual-Stream-PRIME-v4"
)
DELTA_AUDIT_REPORT = (
    ROOT / "runs" / "lfm25-2p6b-heretic-delta-audit" / "report.json"
)

CAPABILITY_RANKS = (1, 2, 4)
GLOBAL_DIRECTION_INDEX = 16.76666648576413
METRIC_REGULARIZATION = 1e-3
KL_HARD_CAP = 0.0142
DEVELOPMENT_KL_TARGET = 0.012
CALIBRATION_STEPS = 7
MAXIMUM_BETA = 4.0
PRIMARY_K = (52,)
GEOMETRIES = ("global_raw", "global_r1", "global_r2", "global_r4")


def prompts(rows: Any, bounds: tuple[int, int]) -> list[str]:
    start, stop = bounds
    return [str(rows[index]["text"]) for index in range(start, stop)]


def capability_metric(result: Any, sample_count: int) -> LowRankMetric:
    covariance_factor = result.capability_basis * (
        result.capability_singular_values
        / math.sqrt(max(sample_count - 1, 1))
    )[None, :]
    return LowRankMetric.from_factors(
        result.capability_basis.shape[0],
        covariance_factor=covariance_factor,
        regularization=METRIC_REGULARIZATION,
    )


def operator_efficiency(
    operator: Any,
    safe: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, float]:
    safe_values = safe.float()
    target_values = target.float()
    centered = safe_values - safe_values.mean(dim=0)
    a = operator.a.float()
    b = operator.b.float()
    safe_delta = (centered @ b) @ a.T
    target_shift = target_values.mean(dim=0) - safe_values.mean(dim=0)
    target_delta = (target_shift @ b) @ a.T
    safe_rms = float(safe_delta.square().sum(dim=1).mean().sqrt())
    target_effect = float(torch.linalg.vector_norm(target_delta))
    return {
        "safe_projection_rms": safe_rms,
        "target_effect": target_effect,
        "efficiency": target_effect / max(safe_rms, 1e-7),
        "operator_spectral_norm": operator.spectral_norm(),
    }


def build_portfolios(
    raw_editors: tuple[Any, ...],
    safe: torch.Tensor,
    target: torch.Tensor,
) -> tuple[dict[str, tuple[Any, ...]], dict[str, list[dict[str, Any]]], Any]:
    axes = fit_residual_stream_axes(
        safe,
        target,
        folds=FOLDS,
        remove_safe_mean=True,
    )
    lower = int(math.floor(GLOBAL_DIRECTION_INDEX))
    fraction = GLOBAL_DIRECTION_INDEX - lower
    if lower < 0 or lower + 1 >= len(axes):
        raise RuntimeError("global residual direction index is out of range")
    global_axis = F.normalize(
        axes[lower].axis.float().lerp(axes[lower + 1].axis.float(), fraction),
        dim=0,
    )
    safe_global = safe[:, lower].float().lerp(safe[:, lower + 1].float(), fraction)
    target_global = target[:, lower].float().lerp(
        target[:, lower + 1].float(), fraction
    )
    safe_mean = safe_global.mean(dim=0)
    global_evidence = ContrastiveAxis(
        axis=global_axis,
        fold_cosine_minimum=min(
            axes[lower].fold_cosine_minimum,
            axes[lower + 1].fold_cosine_minimum,
        ),
        fold_cosine_mean=(
            axes[lower].fold_cosine_mean + axes[lower + 1].fold_cosine_mean
        )
        / 2,
        safe_mean_cosine=float(
            torch.dot(global_axis, F.normalize(safe_mean, dim=0))
        ),
        folds=axes[lower].folds,
    )

    def bind(evidence: ContrastiveAxis) -> tuple[Any, ...]:
        column = evidence.axis[:, None]
        return tuple(
            replace(
                editor,
                operator=replace(editor.operator, a=column, b=column),
                evidence=evidence,
            )
            for editor in raw_editors
        )

    portfolios = {"global_raw": bind(global_evidence)}
    protection_diagnostics = {}
    for capability_rank in CAPABILITY_RANKS:
        protected = protect_residual_stream_axes(
            safe_global[:, None, :],
            target_global[:, None, :],
            (global_evidence,),
            capability_rank=capability_rank,
            seed=2600,
            device="cuda",
        )
        name = f"global_r{capability_rank}"
        portfolios[name] = bind(protected[0].evidence)
        protection_diagnostics[name] = [
            {
                "layer": GLOBAL_DIRECTION_INDEX,
                "retained_fraction": result.retained_fraction,
                "safe_projection_rms": result.safe_projection_rms,
                "target_separation": result.target_separation,
                "efficiency": result.efficiency,
            }
            for layer, result in enumerate(protected)
        ]

    layer_diagnostics = {}
    for geometry, editors in portfolios.items():
        operator_by_layer = {
            editor.site.layer: editor.operator for editor in editors
        }
        layer_diagnostics[geometry] = {
            layer: operator_efficiency(
                operator,
                safe[:, layer],
                target[:, layer],
            )
            for layer, operator in operator_by_layer.items()
        }

    if not DELTA_AUDIT_REPORT.is_file():
        raise RuntimeError("the frozen competitor delta audit is missing")
    delta_audit = json.loads(DELTA_AUDIT_REPORT.read_text(encoding="utf-8"))
    transferred_profile = {
        (int(row["layer"]), str(row["family"])): float(row["relative_l2"])
        for row in delta_audit["changed_tensors"]
    }
    rankings: dict[str, list[dict[str, Any]]] = {}
    for geometry, editors in portfolios.items():
        rows = []
        for editor in editors:
            kernel = transferred_profile.get(
                (editor.site.layer, editor.site.family), 0.0
            )
            efficiency = layer_diagnostics[geometry][editor.site.layer]["efficiency"]
            rows.append(
                {
                    "site_id": editor.site_id,
                    "layer": editor.site.layer,
                    "family": editor.site.family,
                    "kernel": kernel,
                    "efficiency": efficiency,
                    "routing_score": kernel * efficiency,
                }
            )
        rankings[geometry] = sorted(
            (row for row in rows if row["kernel"] > 0),
            key=lambda row: (-row["routing_score"], row["site_id"]),
        )
    diagnostics = {
        "global_direction_index": GLOBAL_DIRECTION_INDEX,
        "transferred_profile_report": str(DELTA_AUDIT_REPORT),
        "transferred_profile_report_sha256": sha256_file(DELTA_AUDIT_REPORT),
        "protected_layers": protection_diagnostics,
        "operator_layers": layer_diagnostics,
    }
    return portfolios, rankings, diagnostics


def calibrated_beta(
    model: Any,
    tokenizer: Any,
    portfolio: tuple[Any, ...],
    originals: dict[str, torch.Tensor],
    ranking: list[dict[str, Any]],
    good_prompts: list[str],
    baseline_log_probs: list[torch.Tensor],
    *,
    k: int,
    target_kl: float = DEVELOPMENT_KL_TARGET,
) -> tuple[float, float]:
    """Find the strongest portfolio that remains below a measured KL target."""

    def probe(beta: float) -> float:
        strengths = candidate_strengths(ranking, k=k, beta=beta)
        apply_residual_stream_weight_edits(model, portfolio, originals, strengths)
        return mean_first_token_kl(
            baseline_log_probs,
            next_token_log_probs(model, tokenizer, good_prompts),
        )

    low = 0.0
    low_kl = 0.0
    high = 0.5
    high_kl = probe(high)
    while high < MAXIMUM_BETA and high_kl < target_kl:
        low, low_kl = high, high_kl
        high = min(high * 2.0, MAXIMUM_BETA)
        high_kl = probe(high)
    if high_kl < target_kl:
        return high, high_kl
    for _ in range(CALIBRATION_STEPS):
        middle = (low + high) / 2.0
        middle_kl = probe(middle)
        if middle_kl < target_kl:
            low, low_kl = middle, middle_kl
        else:
            high = middle
    return low, low_kl


def candidate_strengths(
    ranking: list[dict[str, Any]],
    *,
    k: int,
    beta: float,
) -> dict[str, float]:
    selected = ranking[: min(k, len(ranking))]
    maximum_kernel = max((row["kernel"] for row in selected), default=0.0)
    if maximum_kernel <= 0:
        raise RuntimeError("candidate portfolio has no positive kernel strength")
    return {
        row["site_id"]: beta * row["kernel"] / maximum_kernel
        for row in selected
    }


def trial_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["refusal_markers"],
        row["first_token_kl"],
        row["active_sites"],
        row["geometry"],
        row["beta"],
    )


def evaluate_trial(
    model: Any,
    tokenizer: Any,
    portfolio: tuple[Any, ...],
    originals: dict[str, torch.Tensor],
    ranking: list[dict[str, Any]],
    bad_prompts: list[str],
    good_prompts: list[str],
    baseline_log_probs: list[torch.Tensor],
    *,
    geometry: str,
    k: int,
    beta: float,
    label: str,
) -> dict[str, Any]:
    strengths = candidate_strengths(ranking, k=k, beta=beta)
    apply_residual_stream_weight_edits(model, portfolio, originals, strengths)
    started = time.time()
    refusal = refusal_evaluation(
        model,
        tokenizer,
        bad_prompts,
        label=label,
    )
    kl = mean_first_token_kl(
        baseline_log_probs,
        next_token_log_probs(model, tokenizer, good_prompts),
    )
    result = {
        "geometry": geometry,
        "k": k,
        "beta": beta,
        "active_sites": len(strengths),
        "maximum_strength": max(strengths.values()),
        "refusal_markers": refusal["refusal_markers"],
        "marker_hits": refusal["marker_hits"],
        "response_sha256": refusal["response_sha256"],
        "first_token_kl": kl,
        "seconds": time.time() - started,
    }
    print(json.dumps({"trial": label, **result}), flush=True)
    return result


def main() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    if OUTPUT_PATH.exists():
        raise RuntimeError(f"refusing to overwrite existing output: {OUTPUT_PATH}")
    if not PARENT_REPORT.is_file():
        raise RuntimeError("the frozen Residual-Stream evidence is missing")
    parent = json.loads(PARENT_REPORT.read_text(encoding="utf-8"))
    if parent["source"]["model_id"] != MODEL_ID:
        raise RuntimeError("parent report source model mismatch")
    if parent["source"]["revision"] != MODEL_REVISION:
        raise RuntimeError("parent report source revision mismatch")

    source = base_path()
    tokenizer = AutoTokenizer.from_pretrained(source)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        source,
        dtype=torch.bfloat16,
        device_map=0,
    ).eval()
    if not RESIDUAL_CACHE.is_file():
        good_train = load_dataset(
            GOOD_DATASET,
            revision=GOOD_REVISION,
            split=f"train[:{COLLECT_COUNT}]",
        )
        bad_train = load_dataset(
            BAD_DATASET,
            revision=BAD_REVISION,
            split=f"train[:{COLLECT_COUNT}]",
        )
        safe_residuals = collect_residuals(
            model,
            tokenizer,
            [str(row["text"]) for row in good_train],
            "safe:closed_think",
            close_think=True,
        )
        target_residuals = collect_residuals(
            model,
            tokenizer,
            [str(row["text"]) for row in bad_train],
            "target:closed_think",
            close_think=True,
        )
        save_file(
            {
                "safe": safe_residuals.contiguous(),
                "target": target_residuals.contiguous(),
            },
            RESIDUAL_CACHE,
            metadata={
                "model_revision": MODEL_REVISION,
                "good_revision": GOOD_REVISION,
                "bad_revision": BAD_REVISION,
                "temporal_position": "closed_think_response_start",
            },
        )
    cached = load_file(RESIDUAL_CACHE)
    safe = cached["safe"]
    target = cached["target"]

    registry, _axes, raw_editors = build_editors(model, safe, target)
    portfolios, rankings, geometry_diagnostics = build_portfolios(
        raw_editors,
        safe,
        target,
    )
    originals = snapshot_residual_stream_weights(model, raw_editors)

    bad_test = load_dataset(BAD_DATASET, revision=BAD_REVISION, split="test")
    good_test = load_dataset(GOOD_DATASET, revision=GOOD_REVISION, split="test")
    bad_a = prompts(bad_test, FOLD_A)
    bad_b = prompts(bad_test, FOLD_B)
    bad_holdout = prompts(bad_test, HOLDOUT)
    good_a = prompts(good_test, FOLD_A)
    good_b = prompts(good_test, FOLD_B)
    good_holdout = prompts(good_test, HOLDOUT)

    apply_residual_stream_weight_edits(model, raw_editors, originals, {})
    baseline = {
        "fold_a": refusal_evaluation(model, tokenizer, bad_a, label="base:fold_a"),
        "fold_b": refusal_evaluation(model, tokenizer, bad_b, label="base:fold_b"),
        "holdout": refusal_evaluation(
            model,
            tokenizer,
            bad_holdout,
            label="base:holdout",
        ),
    }
    base_log_probs = {
        "fold_a": next_token_log_probs(model, tokenizer, good_a),
        "fold_b": next_token_log_probs(model, tokenizer, good_b),
        "holdout": next_token_log_probs(model, tokenizer, good_holdout),
    }

    development_bad = bad_a + bad_b
    development_good = good_a + good_b
    development_base_log_probs = (
        base_log_probs["fold_a"] + base_log_probs["fold_b"]
    )

    trials = []
    for geometry in GEOMETRIES:
        for requested_k in PRIMARY_K:
            k = min(requested_k, len(rankings[geometry]))
            if any(row["geometry"] == geometry and row["k"] == k for row in trials):
                continue
            beta, calibrated_kl = calibrated_beta(
                model,
                tokenizer,
                portfolios[geometry],
                originals,
                rankings[geometry],
                development_good,
                development_base_log_probs,
                k=k,
            )
            trial = evaluate_trial(
                model,
                tokenizer,
                portfolios[geometry],
                originals,
                rankings[geometry],
                development_bad,
                development_good,
                development_base_log_probs,
                geometry=geometry,
                k=k,
                beta=beta,
                label=f"development:{geometry}:k{k}:b{beta:.6f}",
            )
            trial["calibrated_first_token_kl"] = calibrated_kl
            trials.append(trial)

    feasible = [
        row
        for row in trials
        if math.isfinite(row["first_token_kl"])
        and row["first_token_kl"] < KL_HARD_CAP
    ]
    if not feasible:
        raise RuntimeError("no Residual-Stream PRIME trial met the hard KL cap")
    feasible.sort(key=trial_key)
    selected = feasible[0]
    holdout = evaluate_trial(
        model,
        tokenizer,
        portfolios[selected["geometry"]],
        originals,
        rankings[selected["geometry"]],
        bad_holdout,
        good_holdout,
        base_log_probs["holdout"],
        geometry=selected["geometry"],
        k=selected["k"],
        beta=selected["beta"],
        label="selected:locked_holdout",
    )
    if holdout["first_token_kl"] >= KL_HARD_CAP:
        raise RuntimeError(
            "selected candidate exceeded the KL cap on the locked holdout"
        )

    model.save_pretrained(
        OUTPUT_PATH,
        safe_serialization=True,
        max_shard_size="10GB",
    )
    tokenizer.save_pretrained(OUTPUT_PATH)
    for filename in ("LICENSE", "chat_template.jinja"):
        source_file = source / filename
        if source_file.exists():
            shutil.copy2(source_file, OUTPUT_PATH / filename)
    output_model = OUTPUT_PATH / "model.safetensors"
    if not output_model.is_file():
        raise RuntimeError("expected a single BF16 output shard")

    report = {
        "schema_version": "lfm25-2p6b-residual-stream-prime-v4",
        "engine": "Heretic NX",
        "algorithm_profile": "Residual-Stream PRIME",
        "source": {
            "model_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "weight_sha256": parent["source"]["weight_sha256"],
        },
        "parent": {
            "report": str(PARENT_REPORT),
            "report_sha256": sha256_file(PARENT_REPORT),
            "residual_cache_sha256": sha256_file(RESIDUAL_CACHE),
        },
        "objective": {
            "hard_first_token_kl_cap": KL_HARD_CAP,
            "development_first_token_kl_target": DEVELOPMENT_KL_TARGET,
            "target_refusals_per_100": 4.0,
            "selection": "minimum paired development refusals under KL cap",
        },
        "method": {
            "capability_ranks": CAPABILITY_RANKS,
            "global_direction_index": GLOBAL_DIRECTION_INDEX,
            "temporal_position": "closed_think_response_start",
            "geometries": GEOMETRIES,
            "primary_k": PRIMARY_K,
            "calibration_steps": CALIBRATION_STEPS,
            "maximum_beta": MAXIMUM_BETA,
            "rankings": rankings,
            "geometry_diagnostics": geometry_diagnostics,
        },
        "baseline": baseline,
        "trials": trials,
        "selected": selected,
        "locked_holdout": holdout,
        "output": {
            "path": str(OUTPUT_PATH),
            "dtype": "bfloat16",
            "quantized": False,
            "model_sha256": sha256_file(output_model),
        },
        "interpretation_guard": (
            "Lexical refusal markers and first-token KL are selection proxies; "
            "XSTest and paired capability are independent promotion gates."
        ),
    }
    REPORT.write_bytes(canonical_json(report) + b"\n")
    (OUTPUT_PATH / "HERETIC_NX_BUILD.json").write_bytes(
        canonical_json(
            {
                "schema_version": report["schema_version"],
                "selected": selected,
                "locked_holdout": holdout,
                "report_sha256": sha256_file(REPORT),
                "model_sha256": report["output"]["model_sha256"],
            }
        )
        + b"\n"
    )
    print(
        json.dumps(
            {
                "selected": selected,
                "holdout": holdout,
                "output": report["output"],
                "report": str(REPORT),
            },
            indent=2,
        ),
        flush=True,
    )
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
