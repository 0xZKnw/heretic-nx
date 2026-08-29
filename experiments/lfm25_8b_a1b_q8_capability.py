#!/usr/bin/env python3
"""Matched capability evaluation for base and Heretic LFM2.5-8B-A1B GGUFs."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
from pathlib import Path
import time
from typing import Any, Callable

from transformers import AutoTokenizer

from experiments.lfm25_2p6b_residual_stream import render
from experiments.lfm25_closed_track_eval import expanded_capability_rows
from experiments.lfm25_residual_stream_capability import LETTERS, task_scores
from heretic_nx.eval.capability import paired_bootstrap_interval
from heretic_nx.eval.gguf_runtime import RESTRICTED_GRAMMAR, native_restricted_choice
from heretic_nx.hashing import canonical_json, sha256_file, sha256_json


ROOT = Path(__file__).resolve().parents[1]
TOKENIZER_PATH = ROOT / "checkpoints" / "lfm25-8b-a1b"
RUN_DIR = ROOT / "runs" / "lfm25-8b-a1b-q8-capability"
RUNTIME = {
    "engine": "llama.cpp",
    "distribution": "LM Studio llama.cpp macOS Metal 2.31.2",
    "build": "0.3.0-dev build 1 commit 1844325",
    "api": "native /completion",
    "parallel_slots": 4,
    "total_context": 4096,
}
BATCH_SIZE = 32
NONINFERIORITY_MARGIN = 0.03
ALPHA = 0.05
RESAMPLES = 10_000
SEED = 8259


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical_json(value) + b"\n")
    temporary.replace(path)


def resume_map(
    *,
    prompts: list[list[int]],
    worker: Callable[[list[int]], str],
    partial_path: Path,
    expected: dict[str, Any],
    parallel: int,
) -> tuple[list[str], float]:
    checkpoint: dict[str, Any] = {
        **expected,
        "completed": 0,
        "values": [],
        "seconds": 0.0,
    }
    if partial_path.is_file():
        loaded = json.loads(partial_path.read_text(encoding="utf-8"))
        if not all(loaded.get(key) == value for key, value in expected.items()):
            raise RuntimeError(f"stale capability checkpoint: {partial_path}")
        checkpoint = loaded
    completed = int(checkpoint["completed"])
    values = [str(value) for value in checkpoint["values"]]
    if completed != len(values) or completed > len(prompts):
        raise RuntimeError(f"invalid capability checkpoint: {partial_path}")
    if completed == len(prompts):
        return values, float(checkpoint["seconds"])

    with ThreadPoolExecutor(max_workers=parallel) as pool:
        for start in range(completed, len(prompts), BATCH_SIZE):
            batch = prompts[start : start + BATCH_SIZE]
            started = time.time()
            produced = list(pool.map(worker, batch))
            checkpoint["seconds"] = float(checkpoint["seconds"]) + (
                time.time() - started
            )
            values.extend(str(value).strip() for value in produced)
            checkpoint["values"] = values
            checkpoint["completed"] = len(values)
            write_json(partial_path, checkpoint)
            print(
                json.dumps(
                    {
                        "capability": expected["label"],
                        "completed": len(values),
                        "total": len(prompts),
                        "seconds": round(float(checkpoint["seconds"]), 3),
                    }
                ),
                flush=True,
            )
    return values, float(checkpoint["seconds"])


def collect(args: argparse.Namespace) -> None:
    artifact = args.artifact.expanduser().resolve(strict=True)
    rows = expanded_capability_rows()
    rows_hash = sha256_json(rows)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    letter_ids = [
        tokenizer.encode(letter, add_special_tokens=False) for letter in LETTERS
    ]
    if any(len(ids) != 1 for ids in letter_ids):
        raise RuntimeError(f"answer labels are not single tokens: {letter_ids}")
    answer_ids = [int(ids[0]) for ids in letter_ids]
    rendered = render(
        tokenizer,
        [str(row["prompt"]) for row in rows],
        close_think=True,
    )
    prompts = [
        [int(token) for token in tokenizer.encode(value, add_special_tokens=False)]
        for value in rendered
    ]
    prompts_hash = sha256_json(prompts)
    artifact_hash = sha256_file(artifact)
    expected = {
        "schema_version": "lfm25-8b-a1b-q8-capability-partial-v1",
        "label": args.label,
        "model": args.model,
        "artifact_sha256": artifact_hash,
        "rows_sha256": rows_hash,
        "prompt_tokens_sha256": prompts_hash,
        "answer_token_ids": answer_ids,
        "grammar": RESTRICTED_GRAMMAR,
        "temperature": -1,
        "max_new_tokens": 1,
        "runtime": RUNTIME,
    }
    partial = RUN_DIR / f"{args.label}.partial.json"
    choices, _seconds = resume_map(
        prompts=prompts,
        worker=lambda tokens: native_restricted_choice(args.endpoint, tokens),
        partial_path=partial,
        expected=expected,
        parallel=args.parallel,
    )
    invalid = [
        {"index": index, "value": value}
        for index, value in enumerate(choices)
        if value not in LETTERS
    ]
    if invalid:
        raise RuntimeError(f"restricted generation escaped answer labels: {invalid[:5]}")
    predictions = [LETTERS.index(choice) for choice in choices]
    correctness = [
        int(prediction == int(row["answer"]))
        for prediction, row in zip(predictions, rows)
    ]
    report = {
        "schema_version": "lfm25-8b-a1b-q8-capability-arm-v1",
        "label": args.label,
        "model": args.model,
        "artifact": {
            "filename": artifact.name,
            "sha256": artifact_hash,
            "size_bytes": artifact.stat().st_size,
        },
        "datasets": {
            "rows": len(rows),
            "rows_sha256": rows_hash,
            "tasks": sorted({str(row["task"]) for row in rows}),
        },
        "protocol": {
            "scoring": "restricted first-token A/B/C/D raw-logit argmax",
            "prompt_tokens_sha256": prompts_hash,
            "answer_token_ids": answer_ids,
            "grammar": RESTRICTED_GRAMMAR,
            "temperature": -1,
            "max_new_tokens": 1,
            "close_think": True,
        },
        "runtime": {**RUNTIME, "endpoint": args.endpoint},
        "results": {
            "count": len(rows),
            "predictions": predictions,
            "correctness": correctness,
            "accuracy": sum(correctness) / len(correctness),
            "tasks": task_scores(rows, correctness),
        },
        "interpretation_guard": (
            "This deterministic MCQ slice is broader than KL but is not a "
            "comprehensive capability benchmark. Throughput is intentionally not "
            "reported because this protocol was designed for paired accuracy."
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
    for key in ("rows_sha256",):
        if base["datasets"][key] != candidate["datasets"][key]:
            raise RuntimeError(f"capability reports disagree on {key}")
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
        "schema_version": "lfm25-8b-a1b-q8-capability-comparison-v1",
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
            "strictly above zero. Noninferiority uses a predeclared 3-point margin. "
            "Throughput is omitted because an interrupted wall-clock run is not a "
            "controlled speed benchmark."
        ),
    }
    output = RUN_DIR / f"{args.candidate_label}-vs-{args.base_label}.json"
    write_json(output, report)
    print(json.dumps({**report, "report": str(output)}, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    collect_parser = subparsers.add_parser("collect")
    collect_parser.add_argument("--label", required=True)
    collect_parser.add_argument("--model", required=True)
    collect_parser.add_argument("--artifact", type=Path, required=True)
    collect_parser.add_argument("--endpoint", default="http://127.0.0.1:1236")
    collect_parser.add_argument("--parallel", type=int, default=4)
    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--base-label", required=True)
    compare_parser.add_argument("--candidate-label", required=True)
    args = parser.parse_args()
    if args.command == "collect":
        if args.parallel <= 0:
            raise ValueError("parallel must be positive")
        collect(args)
    else:
        compare(args)


if __name__ == "__main__":
    main()
