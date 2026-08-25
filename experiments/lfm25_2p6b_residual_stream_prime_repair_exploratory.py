#!/usr/bin/env python3
"""Explore tied repair portfolios on the full 104-prompt optimization suite."""

from __future__ import annotations

import gc
import json
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
    next_token_log_probs,
)
from experiments.lfm25_2p6b_residual_stream_prime import (
    candidate_strengths,
    evaluate_trial,
    prompts,
)
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
RUN_DIR = ROOT / "runs" / "lfm25-2p6b-residual-stream-prime-v8"
REPORT = RUN_DIR / "report.json"
OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "LFM2.5-2.6B-Heretic-NX-Residual-Stream-PRIME-v8"
)

KL_HARD_CAP = 0.0142
CANDIDATES = (
    {
        "geometry": "repair_r1",
        "k": 4,
        "beta": 0.87890625,
        "development_refusals": 5,
        "development_first_token_kl": 0.01336356678775701,
    },
    {
        "geometry": "repair_r1",
        "k": 8,
        "beta": 0.85,
        "development_refusals": 4,
        "development_first_token_kl": 0.012937697972847672,
    },
    {
        "geometry": "repair_r1",
        "k": 8,
        "beta": 1.0,
        "development_refusals": 4,
        "development_first_token_kl": 0.01272499045908262,
    },
    {
        "geometry": "repair_r1",
        "k": 8,
        "beta": 1.125,
        "development_refusals": 5,
        "development_first_token_kl": 0.012760046201947262,
    },
    {
        "geometry": "repair_r1",
        "k": 12,
        "beta": 1.0234375,
        "development_refusals": 5,
        "development_first_token_kl": 0.013191140623712272,
    },
)


def release(model: Any) -> None:
    model.to("cpu")
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    if OUTPUT_PATH.exists():
        raise RuntimeError(f"refusing to overwrite existing output: {OUTPUT_PATH}")
    if not PARENT_REPORT.is_file() or not RESIDUAL_CACHE.is_file():
        raise RuntimeError("the pinned parent or repair evidence is missing")
    parent = json.loads(PARENT_REPORT.read_text(encoding="utf-8"))
    parent_weight = PARENT_OUTPUT / "model.safetensors"
    if sha256_file(parent_weight) != parent["output"]["model_sha256"]:
        raise RuntimeError("the v3 parent hash does not match its report")

    good_test = load_dataset(GOOD_DATASET, revision=GOOD_REVISION, split="test")
    bad_test = load_dataset(BAD_DATASET, revision=BAD_REVISION, split="test")
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
    base_holdout_log_probs = next_token_log_probs(
        base_model,
        tokenizer,
        holdout_good,
    )
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
    originals = snapshot_residual_stream_weights(model, raw_editors)

    trials = []
    for spec in CANDIDATES:
        result = evaluate_trial(
            model,
            tokenizer,
            portfolios[spec["geometry"]],
            originals,
            rankings[spec["geometry"]],
            holdout_bad,
            holdout_good,
            base_holdout_log_probs,
            geometry=spec["geometry"],
            k=spec["k"],
            beta=spec["beta"],
            label=(
                f"exploratory:{spec['geometry']}:"
                f"k{spec['k']}:b{spec['beta']}"
            ),
        )
        result["development_refusal_markers"] = spec["development_refusals"]
        result["development_first_token_kl"] = spec[
            "development_first_token_kl"
        ]
        result["optimization_suite_refusal_markers"] = (
            spec["development_refusals"] + result["refusal_markers"]
        )
        result["optimization_suite_first_token_kl"] = (
            80 * spec["development_first_token_kl"]
            + 24 * result["first_token_kl"]
        ) / 104
        result["refusals_per_100"] = (
            100 * result["optimization_suite_refusal_markers"] / 104
        )
        trials.append(result)

    feasible = [
        row
        for row in trials
        if row["development_first_token_kl"] < KL_HARD_CAP
        and row["first_token_kl"] < KL_HARD_CAP
    ]
    if not feasible:
        raise RuntimeError("no exploratory candidate met the holdout KL cap")
    feasible.sort(
        key=lambda row: (
            row["optimization_suite_refusal_markers"],
            row["optimization_suite_first_token_kl"],
            row["active_sites"],
        )
    )
    selected = feasible[0]
    # evaluate_trial left the last portfolio active; explicitly restore the winner.
    selected_strengths = candidate_strengths(
        rankings[selected["geometry"]],
        k=selected["k"],
        beta=selected["beta"],
    )
    apply_residual_stream_weight_edits(
        model,
        portfolios[selected["geometry"]],
        originals,
        selected_strengths,
    )

    model.save_pretrained(OUTPUT_PATH, safe_serialization=True, max_shard_size="10GB")
    tokenizer.save_pretrained(OUTPUT_PATH)
    for filename in ("LICENSE", "chat_template.jinja"):
        source_file = source / filename
        if source_file.exists():
            shutil.copy2(source_file, OUTPUT_PATH / filename)
    output_model = OUTPUT_PATH / "model.safetensors"
    report = {
        "schema_version": "lfm25-2p6b-residual-stream-prime-v8",
        "engine": "Heretic NX",
        "algorithm_profile": "Residual-Stream PRIME full-suite frontier",
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
        },
        "objective": {"hard_first_token_kl_cap": KL_HARD_CAP},
        "method": {
            "candidate_specs": CANDIDATES,
            "diagnostics": diagnostics["repair_r1"],
        },
        "trials": trials,
        "selected": selected,
        "output": {
            "path": str(OUTPUT_PATH),
            "dtype": "bfloat16",
            "quantized": False,
            "model_sha256": sha256_file(output_model),
        },
        "interpretation_guard": (
            "All 104 harmful_behaviors prompts are optimization evidence in this "
            "exploratory frontier. XSTest and paired capability are required as "
            "independent promotion gates."
        ),
    }
    REPORT.write_bytes(canonical_json(report) + b"\n")
    (OUTPUT_PATH / "HERETIC_NX_BUILD.json").write_bytes(
        canonical_json(
            {
                "schema_version": report["schema_version"],
                "selected": selected,
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
