#!/usr/bin/env python3
"""Matched raw-logit capability evaluation for Ling-3.0-tiny Q8 GGUFs."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import numpy as np
from transformers import AutoTokenizer

from experiments.lfm25_closed_track_eval import expanded_capability_rows
from experiments.lfm25_residual_stream_capability import LETTERS, task_scores
from experiments.ling3_tiny_q8_eval import render
from heretic_nx.eval.capability import paired_bootstrap_interval
from heretic_nx.hashing import canonical_json, sha256_file, sha256_json


ROOT = Path(__file__).resolve().parents[1]
TOKENIZER_PATH = ROOT / "checkpoints" / "ling3-tiny"
RUN_DIR = ROOT / "runs" / "ling3-tiny-q8-capability"
NONINFERIORITY_MARGIN = 0.03
ALPHA = 0.05
RESAMPLES = 10_000
SEED = 8259
RUNTIME = {
    "engine": "llama.cpp",
    "distribution": "LM Studio llama.cpp macOS Metal 2.31.2",
    "build": "commit 18443257a30c884d5332abb8e7dc43c7ffe42fda",
    "api": "native C ABI raw logits",
    "gpu_layers": "all",
    "context": 1024,
}


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical_json(value) + b"\n")
    temporary.replace(path)


def evaluation_inputs() -> tuple[list[dict[str, Any]], list[list[int]], list[int]]:
    rows = expanded_capability_rows()
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
    answer_rows = [
        tokenizer.encode(letter, add_special_tokens=False) for letter in LETTERS
    ]
    if any(len(token_ids) != 1 for token_ids in answer_rows):
        raise RuntimeError(f"answer labels are not single tokens: {answer_rows}")
    answer_ids = [int(token_ids[0]) for token_ids in answer_rows]
    rendered = render(tokenizer, [str(row["prompt"]) for row in rows])
    prompts = [
        [int(token) for token in tokenizer.encode(value, add_special_tokens=False)]
        for value in rendered
    ]
    return rows, prompts, answer_ids


def export_tokens(args: argparse.Namespace) -> None:
    rows, prompts, answer_ids = evaluation_inputs()
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(" ".join(str(token) for token in row) + "\n" for row in prompts),
        encoding="utf-8",
    )
    metadata = {
        "schema_version": "ling3-tiny-q8-capability-prompts-v1",
        "count": len(rows),
        "rows_sha256": sha256_json(rows),
        "prompt_tokens_sha256": sha256_json(prompts),
        "answer_token_ids": answer_ids,
        "thinking": "off",
        "minimum_prompt_tokens": min(map(len, prompts)),
        "maximum_prompt_tokens": max(map(len, prompts)),
        "output": str(output),
    }
    write_json(output.with_suffix(output.suffix + ".json"), metadata)
    print(json.dumps(metadata, indent=2), flush=True)


def score_raw(args: argparse.Namespace) -> None:
    rows, prompts, answer_ids = evaluation_inputs()
    artifact = args.artifact.expanduser().resolve(strict=True)
    raw = args.raw.expanduser().resolve(strict=True)
    expected_bytes = len(rows) * len(answer_ids) * np.dtype(np.float32).itemsize
    if raw.stat().st_size != expected_bytes:
        raise RuntimeError(
            f"invalid selected-logit file size: {raw.stat().st_size} != {expected_bytes}"
        )
    logits = np.memmap(
        raw,
        mode="r",
        dtype=np.float32,
        shape=(len(rows), len(answer_ids)),
    )
    if not np.isfinite(logits).all():
        raise RuntimeError("selected logits contain non-finite values")
    predictions = np.argmax(logits, axis=1).astype(int).tolist()
    correctness = [
        int(prediction == int(row["answer"]))
        for prediction, row in zip(predictions, rows)
    ]
    report = {
        "schema_version": "ling3-tiny-q8-capability-arm-v1",
        "label": args.label,
        "model": args.model,
        "artifact": {
            "filename": artifact.name,
            "sha256": sha256_file(artifact),
            "size_bytes": artifact.stat().st_size,
        },
        "raw_selected_logits": {
            "filename": raw.name,
            "sha256": sha256_file(raw),
            "shape": [len(rows), len(answer_ids)],
        },
        "datasets": {
            "rows": len(rows),
            "rows_sha256": sha256_json(rows),
            "tasks": sorted({str(row["task"]) for row in rows}),
        },
        "protocol": {
            "scoring": "restricted first-token A/B/C/D raw-logit argmax",
            "prompt_tokens_sha256": sha256_json(prompts),
            "answer_token_ids": answer_ids,
            "thinking": "off",
            "temperature": -1,
            "max_new_tokens": 1,
        },
        "runtime": RUNTIME,
        "results": {
            "count": len(rows),
            "predictions": predictions,
            "correctness": correctness,
            "accuracy": sum(correctness) / len(correctness),
            "tasks": task_scores(rows, correctness),
        },
        "interpretation_guard": (
            "This deterministic MCQ slice is broader than KL but is not a "
            "comprehensive capability benchmark."
        ),
    }
    output = RUN_DIR / f"{args.label}.json"
    write_json(output, report)
    print(
        json.dumps(
            {
                "label": args.label,
                "accuracy": report["results"]["accuracy"],
                "tasks": report["results"]["tasks"],
                "report": str(output),
            },
            indent=2,
        ),
        flush=True,
    )


def compare(args: argparse.Namespace) -> None:
    base_path = RUN_DIR / f"{args.base_label}.json"
    candidate_path = RUN_DIR / f"{args.candidate_label}.json"
    base = json.loads(base_path.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    if base["datasets"]["rows_sha256"] != candidate["datasets"]["rows_sha256"]:
        raise RuntimeError("capability reports used different rows")
    if (
        base["protocol"]["prompt_tokens_sha256"]
        != candidate["protocol"]["prompt_tokens_sha256"]
    ):
        raise RuntimeError("capability reports used different prompt tokens")
    base_correctness = [int(value) for value in base["results"]["correctness"]]
    candidate_correctness = [
        int(value) for value in candidate["results"]["correctness"]
    ]
    interval = asdict(
        paired_bootstrap_interval(
            base_correctness,
            candidate_correctness,
            margin=NONINFERIORITY_MARGIN,
            alpha=ALPHA,
            resamples=RESAMPLES,
            seed=SEED,
        )
    )
    paired_counts = {
        "both_correct": sum(
            left == 1 and right == 1
            for left, right in zip(base_correctness, candidate_correctness)
        ),
        "base_only_correct": sum(
            left == 1 and right == 0
            for left, right in zip(base_correctness, candidate_correctness)
        ),
        "candidate_only_correct": sum(
            left == 0 and right == 1
            for left, right in zip(base_correctness, candidate_correctness)
        ),
        "both_incorrect": sum(
            left == 0 and right == 0
            for left, right in zip(base_correctness, candidate_correctness)
        ),
    }
    tasks = {}
    for task in sorted(base["results"]["tasks"]):
        base_task = base["results"]["tasks"][task]
        candidate_task = candidate["results"]["tasks"][task]
        tasks[task] = {
            "count": base_task["count"],
            "base_accuracy": base_task["accuracy"],
            "candidate_accuracy": candidate_task["accuracy"],
            "difference": candidate_task["accuracy"] - base_task["accuracy"],
        }
    report = {
        "schema_version": "ling3-tiny-q8-capability-comparison-v1",
        "base_label": args.base_label,
        "candidate_label": args.candidate_label,
        "rows": len(base_correctness),
        "rows_sha256": base["datasets"]["rows_sha256"],
        "protocol": base["protocol"],
        "artifacts": {
            "base": base["artifact"],
            "candidate": candidate["artifact"],
        },
        "results": {
            "base_accuracy": base["results"]["accuracy"],
            "candidate_accuracy": candidate["results"]["accuracy"],
            "candidate_minus_base": interval,
            "paired_counts": paired_counts,
            "tasks": tasks,
        },
        "passed_noninferiority": interval["noninferiority_passed"],
        "demonstrated_accuracy_improvement": interval["lower"] > 0.0,
        "interpretation_guard": (
            "Accuracy improvement is claimed only when the paired 95% interval is "
            "strictly above zero. Noninferiority uses a predeclared 3-point margin."
        ),
    }
    output = RUN_DIR / f"{args.candidate_label}-vs-{args.base_label}.json"
    write_json(output, report)
    print(json.dumps({**report, "report": str(output)}, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    export_parser = subparsers.add_parser("export-tokens")
    export_parser.add_argument("--output", type=Path, required=True)
    score_parser = subparsers.add_parser("score-raw")
    score_parser.add_argument("--label", required=True)
    score_parser.add_argument("--model", required=True)
    score_parser.add_argument("--artifact", type=Path, required=True)
    score_parser.add_argument("--raw", type=Path, required=True)
    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--base-label", required=True)
    compare_parser.add_argument("--candidate-label", required=True)
    args = parser.parse_args()
    if args.command == "export-tokens":
        export_tokens(args)
    elif args.command == "score-raw":
        score_raw(args)
    else:
        compare(args)


if __name__ == "__main__":
    main()
