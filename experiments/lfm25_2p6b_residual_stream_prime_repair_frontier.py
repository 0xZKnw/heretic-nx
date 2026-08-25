#!/usr/bin/env python3
"""Fine intensity frontier for the first iterative 2.6B repair axis."""

from __future__ import annotations

import gc
import json
import math
from pathlib import Path
import shutil
from typing import Any

from datasets import load_dataset
from safetensors.torch import load_file
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.lfm25_2p6b_residual_stream import (
    BAD_DATASET,
    BAD_REVISION,
    GOOD_DATASET,
    GOOD_REVISION,
    HOLDOUT,
    MODEL_ID,
    MODEL_REVISION,
    base_path,
    build_editors,
    mean_first_token_kl,
    next_token_log_probs,
    refusal_evaluation,
)
from experiments.lfm25_2p6b_residual_stream_prime import evaluate_trial, prompts
from experiments.lfm25_2p6b_residual_stream_prime_repair import (
    build_repair_portfolios,
)
from heretic_nx.edits.residual_stream import (
    apply_residual_stream_weight_edits,
    snapshot_residual_stream_weights,
)
from heretic_nx.hashing import canonical_json, sha256_file


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
PARENT_OUTPUT = (
    PROJECT_ROOT
    / "outputs"
    / "LFM2.5-2.6B-Heretic-NX-Residual-Stream-PRIME-v3"
)
PARENT_REPORT = (
    ROOT / "runs" / "lfm25-2p6b-residual-stream-prime-v3" / "report.json"
)
EVIDENCE_DIR = ROOT / "runs" / "lfm25-2p6b-residual-stream-prime-v5"
RESIDUAL_CACHE = EVIDENCE_DIR / "repair-residuals.safetensors"
SCREENING_CACHE = EVIDENCE_DIR / "training-screening.json"
RUN_DIR = ROOT / "runs" / "lfm25-2p6b-residual-stream-prime-v7"
REPORT = RUN_DIR / "report.json"
OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "LFM2.5-2.6B-Heretic-NX-Residual-Stream-PRIME-v7"
)

DEVELOPMENT = (0, 80)
KL_HARD_CAP = 0.0142
GEOMETRY = "repair_r1"
PORTFOLIO_K = 8
BETAS = (0.50, 0.70, 0.85, 1.00, 1.125, 1.25, 1.40)


def release(model: Any) -> None:
    model.to("cpu")
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    if OUTPUT_PATH.exists():
        raise RuntimeError(f"refusing to overwrite existing output: {OUTPUT_PATH}")
    required = (PARENT_REPORT, RESIDUAL_CACHE, SCREENING_CACHE)
    if not all(path.is_file() for path in required) or not PARENT_OUTPUT.is_dir():
        raise RuntimeError("the pinned v3 parent or v5 repair evidence is missing")
    parent = json.loads(PARENT_REPORT.read_text(encoding="utf-8"))
    parent_weight = PARENT_OUTPUT / "model.safetensors"
    if sha256_file(parent_weight) != parent["output"]["model_sha256"]:
        raise RuntimeError("the v3 parent hash does not match its report")

    good_test = load_dataset(GOOD_DATASET, revision=GOOD_REVISION, split="test")
    bad_test = load_dataset(BAD_DATASET, revision=BAD_REVISION, split="test")
    development_good = prompts(good_test, DEVELOPMENT)
    development_bad = prompts(bad_test, DEVELOPMENT)
    holdout_good = prompts(good_test, HOLDOUT)
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
    release(base_model)
    del base_model

    model = AutoModelForCausalLM.from_pretrained(
        PARENT_OUTPUT,
        dtype=torch.bfloat16,
        device_map=0,
    ).eval()
    cached = load_file(RESIDUAL_CACHE)
    safe = cached["safe"]
    target = cached["target"]
    _registry, axes, raw_editors = build_editors(model, safe, target, folds=2)
    portfolios, rankings, diagnostics = build_repair_portfolios(
        raw_editors,
        safe,
        target,
        axes,
    )
    portfolio = portfolios[GEOMETRY]
    ranking = rankings[GEOMETRY]
    originals = snapshot_residual_stream_weights(model, raw_editors)

    apply_residual_stream_weight_edits(model, raw_editors, originals, {})
    baseline = {
        "development": refusal_evaluation(
            model,
            tokenizer,
            development_bad,
            label="v3:development",
        ),
        "development_first_token_kl": mean_first_token_kl(
            base_log_probs["development"],
            next_token_log_probs(model, tokenizer, development_good),
        ),
    }

    trials = []
    for beta in BETAS:
        trial = evaluate_trial(
            model,
            tokenizer,
            portfolio,
            originals,
            ranking,
            development_bad,
            development_good,
            base_log_probs["development"],
            geometry=GEOMETRY,
            k=PORTFOLIO_K,
            beta=beta,
            label=f"frontier:{GEOMETRY}:k{PORTFOLIO_K}:b{beta}",
        )
        trials.append(trial)

    feasible = [
        row
        for row in trials
        if math.isfinite(row["first_token_kl"])
        and row["first_token_kl"] < KL_HARD_CAP
    ]
    if not feasible:
        raise RuntimeError("no fine repair trial met the hard KL cap")
    feasible.sort(
        key=lambda row: (
            row["refusal_markers"],
            row["first_token_kl"],
            row["maximum_strength"],
        )
    )
    selected = feasible[0]
    holdout = evaluate_trial(
        model,
        tokenizer,
        portfolio,
        originals,
        ranking,
        holdout_bad,
        holdout_good,
        base_log_probs["holdout"],
        geometry=GEOMETRY,
        k=PORTFOLIO_K,
        beta=selected["beta"],
        label="frontier:selected_locked_holdout",
    )
    if holdout["first_token_kl"] >= KL_HARD_CAP:
        raise RuntimeError("fine repair exceeded the locked holdout KL cap")

    model.save_pretrained(OUTPUT_PATH, safe_serialization=True, max_shard_size="10GB")
    tokenizer.save_pretrained(OUTPUT_PATH)
    for filename in ("LICENSE", "chat_template.jinja"):
        source_file = source / filename
        if source_file.exists():
            shutil.copy2(source_file, OUTPUT_PATH / filename)
    output_model = OUTPUT_PATH / "model.safetensors"
    report = {
        "schema_version": "lfm25-2p6b-residual-stream-prime-v7",
        "engine": "Heretic NX",
        "algorithm_profile": "Residual-Stream PRIME repair frontier",
        "source": {"model_id": MODEL_ID, "revision": MODEL_REVISION},
        "parent": {
            "path": str(PARENT_OUTPUT),
            "report": str(PARENT_REPORT),
            "report_sha256": sha256_file(PARENT_REPORT),
            "model_sha256": sha256_file(parent_weight),
        },
        "repair_evidence": {
            "residual_cache": str(RESIDUAL_CACHE),
            "residual_cache_sha256": sha256_file(RESIDUAL_CACHE),
            "screening_cache": str(SCREENING_CACHE),
            "screening_cache_sha256": sha256_file(SCREENING_CACHE),
        },
        "objective": {"hard_first_token_kl_cap": KL_HARD_CAP},
        "method": {
            "geometry": GEOMETRY,
            "portfolio_k": PORTFOLIO_K,
            "betas": BETAS,
            "diagnostics": diagnostics[GEOMETRY],
            "ranking": ranking,
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
            "Intensity is selected on development prompts only; the holdout is "
            "opened once after selection."
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
    release(model)


if __name__ == "__main__":
    main()
