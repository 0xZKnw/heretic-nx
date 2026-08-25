#!/usr/bin/env python3
"""Iterative residual-boundary repair for the 2.6B PRIME candidate."""

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
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.lfm25_2p6b_residual_stream import (
    BAD_DATASET,
    BAD_REVISION,
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
from experiments.lfm25_2p6b_residual_stream_prime import (
    candidate_strengths,
    evaluate_trial,
    operator_efficiency,
    prompts,
)
from heretic_nx.edits.residual_stream import (
    apply_residual_stream_weight_edits,
    build_residual_stream_weight_editors,
    snapshot_residual_stream_weights,
)
from heretic_nx.geometry.contrastive import ContrastiveAxis, fit_contrastive_axis
from heretic_nx.geometry.residual import protect_residual_stream_axes
from heretic_nx.hashing import canonical_json, sha256_file
from heretic_nx.model import assert_lfm25_layout, discover_semantic_sites


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
PARENT_OUTPUT = (
    PROJECT_ROOT
    / "outputs"
    / "LFM2.5-2.6B-Heretic-NX-Residual-Stream-PRIME-v5"
)
PARENT_REPORT = (
    ROOT / "runs" / "lfm25-2p6b-residual-stream-prime-v5" / "report.json"
)
RUN_DIR = ROOT / "runs" / "lfm25-2p6b-residual-stream-prime-v6"
REPORT = RUN_DIR / "report.json"
OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "LFM2.5-2.6B-Heretic-NX-Residual-Stream-PRIME-v6"
)

TRAIN_COUNT = 400
CAPABILITY_RANKS = (1,)
PORTFOLIO_K = (4, 8, 12)
DEVELOPMENT = (0, 80)
DEVELOPMENT_KL_TARGET = 0.0140
KL_HARD_CAP = 0.0142
CALIBRATION_STEPS = 7
MAXIMUM_REPAIR_BETA = 2.0
SCREENING_CACHE = RUN_DIR / "training-screening.json"
RESIDUAL_CACHE = RUN_DIR / "repair-residuals.safetensors"


def unload(model: Any) -> None:
    model.to("cpu")
    del model
    gc.collect()
    torch.cuda.empty_cache()


def fit_repair_axes(
    safe: torch.Tensor,
    target: torch.Tensor,
    fallback_safe: torch.Tensor,
    fallback_target: torch.Tensor,
) -> tuple[tuple[Any, ...], list[int], list[int]]:
    axes = []
    fallback_layers = []
    invariant_layers = []
    for layer in range(safe.shape[1]):
        try:
            axis = fit_contrastive_axis(
                safe[:, layer],
                target[:, layer],
                folds=2,
                remove_safe_mean=True,
            )
        except RuntimeError:
            try:
                axis = fit_contrastive_axis(
                    safe[:, layer],
                    target[:, layer],
                    folds=2,
                    remove_safe_mean=False,
                )
                fallback_layers.append(layer)
            except RuntimeError:
                prior_direction = (
                    fallback_target[:, layer].float().mean(dim=0)
                    - fallback_safe[:, layer].float().mean(dim=0)
                )
                prior_unit = torch.nn.functional.normalize(prior_direction, dim=0)
                axis = ContrastiveAxis(
                    axis=prior_unit,
                    fold_cosine_minimum=0.0,
                    fold_cosine_mean=0.0,
                    safe_mean_cosine=0.0,
                    folds=2,
                )
                invariant_layers.append(layer)
        axes.append(axis)
    return tuple(axes), fallback_layers, invariant_layers


def build_repair_portfolios(
    raw_editors: tuple[Any, ...],
    safe: torch.Tensor,
    target: torch.Tensor,
    axes: tuple[Any, ...],
) -> tuple[dict[str, tuple[Any, ...]], dict[str, list[dict[str, Any]]], dict[str, Any]]:
    portfolios = {"repair_raw": raw_editors}
    diagnostics: dict[str, Any] = {}
    for rank in CAPABILITY_RANKS:
        protected = protect_residual_stream_axes(
            safe,
            target,
            axes,
            capability_rank=rank,
            seed=2605,
            device="cuda",
        )
        evidence = {
            layer: result.evidence for layer, result in enumerate(protected)
        }
        name = f"repair_r{rank}"
        portfolios[name] = tuple(
            replace(
                editor,
                operator=replace(
                    editor.operator,
                    a=evidence[editor.site.layer].axis[:, None],
                    b=evidence[editor.site.layer].axis[:, None],
                ),
                evidence=evidence[editor.site.layer],
            )
            for editor in raw_editors
        )
        diagnostics[name] = [
            {
                "layer": layer,
                "retained_fraction": result.retained_fraction,
                "safe_projection_rms": result.safe_projection_rms,
                "target_separation": result.target_separation,
                "efficiency": result.efficiency,
            }
            for layer, result in enumerate(protected)
        ]

    rankings = {}
    layer_count = safe.shape[1]
    denominator = max(layer_count - 1, 1)
    for geometry, editors in portfolios.items():
        rows = []
        for editor in editors:
            reference_layer = (
                editor.site.layer * (REFERENCE_LAYER_COUNT - 1) / denominator
            )
            kernel = REFERENCE_KERNELS[editor.site.family].strength(reference_layer)
            efficiency = operator_efficiency(
                editor.operator,
                safe[:, editor.site.layer],
                target[:, editor.site.layer],
            )["efficiency"]
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
    return portfolios, rankings, diagnostics


def calibrate_repair_beta(
    model: Any,
    tokenizer: Any,
    portfolio: tuple[Any, ...],
    originals: dict[str, torch.Tensor],
    ranking: list[dict[str, Any]],
    good_prompts: list[str],
    base_log_probs: list[torch.Tensor],
    *,
    k: int,
) -> tuple[float, float]:
    def probe(beta: float) -> float:
        strengths = candidate_strengths(ranking, k=k, beta=beta)
        apply_residual_stream_weight_edits(model, portfolio, originals, strengths)
        return mean_first_token_kl(
            base_log_probs,
            next_token_log_probs(model, tokenizer, good_prompts),
        )

    low = 0.0
    low_kl = probe(0.0)
    if low_kl >= DEVELOPMENT_KL_TARGET:
        return 0.0, low_kl
    high = 0.0625
    high_kl = probe(high)
    while high < MAXIMUM_REPAIR_BETA and high_kl < DEVELOPMENT_KL_TARGET:
        low, low_kl = high, high_kl
        high = min(high * 2, MAXIMUM_REPAIR_BETA)
        high_kl = probe(high)
    if high_kl < DEVELOPMENT_KL_TARGET:
        return high, high_kl
    for _ in range(CALIBRATION_STEPS):
        middle = (low + high) / 2
        middle_kl = probe(middle)
        if middle_kl < DEVELOPMENT_KL_TARGET:
            low, low_kl = middle, middle_kl
        else:
            high = middle
    return low, low_kl


def main() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    if OUTPUT_PATH.exists():
        raise RuntimeError(f"refusing to overwrite existing output: {OUTPUT_PATH}")
    if not PARENT_REPORT.is_file() or not PARENT_OUTPUT.is_dir():
        raise RuntimeError("the v5 PRIME parent is missing")
    parent = json.loads(PARENT_REPORT.read_text(encoding="utf-8"))
    parent_weight = PARENT_OUTPUT / "model.safetensors"
    if sha256_file(parent_weight) != parent["output"]["model_sha256"]:
        raise RuntimeError("the v5 PRIME parent hash does not match its report")

    good_test = load_dataset(GOOD_DATASET, revision=GOOD_REVISION, split="test")
    bad_test = load_dataset(BAD_DATASET, revision=BAD_REVISION, split="test")
    development_good = prompts(good_test, DEVELOPMENT)
    holdout_good = prompts(good_test, HOLDOUT)
    development_bad = prompts(bad_test, DEVELOPMENT)
    holdout_bad = prompts(bad_test, HOLDOUT)

    source = base_path()
    tokenizer = AutoTokenizer.from_pretrained(source)
    tokenizer.padding_side = "left"
    base_model = AutoModelForCausalLM.from_pretrained(
        source,
        dtype=torch.bfloat16,
        device_map=0,
    ).eval()
    base_log_probs = {
        "development": next_token_log_probs(base_model, tokenizer, development_good),
        "holdout": next_token_log_probs(base_model, tokenizer, holdout_good),
    }
    unload(base_model)
    del base_model
    gc.collect()

    model = AutoModelForCausalLM.from_pretrained(
        PARENT_OUTPUT,
        dtype=torch.bfloat16,
        device_map=0,
    ).eval()
    bad_train = load_dataset(
        BAD_DATASET,
        revision=BAD_REVISION,
        split=f"train[:{TRAIN_COUNT}]",
    )
    good_train = load_dataset(
        GOOD_DATASET,
        revision=GOOD_REVISION,
        split=f"train[:{TRAIN_COUNT}]",
    )
    train_prompts = [str(row["text"]) for row in bad_train]
    if SCREENING_CACHE.is_file():
        screening = json.loads(SCREENING_CACHE.read_text(encoding="utf-8"))
    else:
        screening = refusal_evaluation(
            model,
            tokenizer,
            train_prompts,
            label="v5:training_residual_refusals",
        )
        SCREENING_CACHE.write_bytes(canonical_json(screening) + b"\n")
    residual_targets = [
        prompt
        for prompt, hit in zip(train_prompts, screening["marker_hits"])
        if hit
    ]
    if len(residual_targets) < 12:
        raise RuntimeError("too few residual refusal prompts for a repair direction")
    sample_count = min(len(residual_targets), TRAIN_COUNT)
    if RESIDUAL_CACHE.is_file():
        cached = load_file(RESIDUAL_CACHE)
        safe_residuals = cached["safe"]
        target_residuals = cached["target"]
    else:
        safe_residuals = collect_residuals(
            model,
            tokenizer,
            [str(good_train[index]["text"]) for index in range(sample_count)],
            "repair:safe",
            close_think=True,
        )
        target_residuals = collect_residuals(
            model,
            tokenizer,
            residual_targets[:sample_count],
            "repair:target",
            close_think=True,
        )
        save_file(
            {
                "safe": safe_residuals.contiguous(),
                "target": target_residuals.contiguous(),
            },
            RESIDUAL_CACHE,
            metadata={
                "parent_model_sha256": parent["output"]["model_sha256"],
                "screening_sha256": sha256_file(SCREENING_CACHE),
            },
        )
    parent_residual_cache = (
        PARENT_REPORT.parent.parent
        / "lfm25-2p6b-residual-stream-prime-v5"
        / "repair-residuals.safetensors"
    )
    if not parent_residual_cache.is_file():
        raise RuntimeError("the parent repair residual cache is missing")
    parent_cached = load_file(parent_residual_cache)
    axes, fallback_layers, invariant_layers = fit_repair_axes(
        safe_residuals,
        target_residuals,
        parent_cached["safe"],
        parent_cached["target"],
    )
    registry = discover_semantic_sites(model)
    assert_lfm25_layout(
        registry,
        layer_types=tuple(str(value) for value in model.config.layer_types),
    )
    raw_editors = build_residual_stream_weight_editors(
        registry,
        axes,
        families=frozenset(REFERENCE_KERNELS),
    )
    portfolios, rankings, diagnostics = build_repair_portfolios(
        raw_editors,
        safe_residuals,
        target_residuals,
        axes,
    )
    originals = snapshot_residual_stream_weights(model, raw_editors)

    apply_residual_stream_weight_edits(model, raw_editors, originals, {})
    parent_baseline = {
        "development": refusal_evaluation(
            model,
            tokenizer,
            development_bad,
            label="v5:development",
        ),
        "development_first_token_kl": mean_first_token_kl(
            base_log_probs["development"],
            next_token_log_probs(model, tokenizer, development_good),
        ),
    }

    trials = []
    for geometry, portfolio in portfolios.items():
        for requested_k in PORTFOLIO_K:
            k = min(requested_k, len(rankings[geometry]))
            beta, calibrated_kl = calibrate_repair_beta(
                model,
                tokenizer,
                portfolio,
                originals,
                rankings[geometry],
                development_good,
                base_log_probs["development"],
                k=k,
            )
            trial = evaluate_trial(
                model,
                tokenizer,
                portfolio,
                originals,
                rankings[geometry],
                development_bad,
                development_good,
                base_log_probs["development"],
                geometry=geometry,
                k=k,
                beta=beta,
                label=f"repair:{geometry}:k{k}:b{beta:.6f}",
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
        raise RuntimeError("no residual repair met the hard KL cap")
    feasible.sort(
        key=lambda row: (
            row["refusal_markers"],
            row["first_token_kl"],
            row["active_sites"],
        )
    )
    selected = feasible[0]
    holdout = evaluate_trial(
        model,
        tokenizer,
        portfolios[selected["geometry"]],
        originals,
        rankings[selected["geometry"]],
        holdout_bad,
        holdout_good,
        base_log_probs["holdout"],
        geometry=selected["geometry"],
        k=selected["k"],
        beta=selected["beta"],
        label="repair:selected_locked_holdout",
    )
    if holdout["first_token_kl"] >= KL_HARD_CAP:
        raise RuntimeError("residual repair exceeded the locked holdout KL cap")

    model.save_pretrained(OUTPUT_PATH, safe_serialization=True, max_shard_size="10GB")
    tokenizer.save_pretrained(OUTPUT_PATH)
    for filename in ("LICENSE", "chat_template.jinja"):
        source_file = source / filename
        if source_file.exists():
            shutil.copy2(source_file, OUTPUT_PATH / filename)
    output_model = OUTPUT_PATH / "model.safetensors"
    report = {
        "schema_version": "lfm25-2p6b-residual-stream-prime-v6",
        "engine": "Heretic NX",
        "algorithm_profile": "Residual-Stream PRIME iterative repair",
        "source": {"model_id": MODEL_ID, "revision": MODEL_REVISION},
        "parent": {
            "path": str(PARENT_OUTPUT),
            "report": str(PARENT_REPORT),
            "report_sha256": sha256_file(PARENT_REPORT),
            "model_sha256": sha256_file(parent_weight),
        },
        "objective": {
            "development_first_token_kl_target": DEVELOPMENT_KL_TARGET,
            "hard_first_token_kl_cap": KL_HARD_CAP,
        },
        "screening": screening,
        "repair_sample_count": sample_count,
        "nonorthogonal_fallback_layers": fallback_layers,
        "invariant_layers_using_parent_axis": invariant_layers,
        "parent_baseline": parent_baseline,
        "method": {
            "capability_ranks": CAPABILITY_RANKS,
            "portfolio_k": PORTFOLIO_K,
            "diagnostics": diagnostics,
            "rankings": rankings,
        },
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
            "The repair is selected on development prompts only. XSTest and "
            "paired capability remain independent promotion gates."
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
                "screening_refusals": screening["refusal_markers"],
                "selected": selected,
                "holdout": holdout,
                "output": report["output"],
                "report": str(REPORT),
            },
            indent=2,
        ),
        flush=True,
    )
    unload(model)
    del model


if __name__ == "__main__":
    main()
