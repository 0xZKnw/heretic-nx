#!/usr/bin/env python3
"""Matched 854-row capability evaluation for Gemma 4 E2B Q8 artifacts."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
from pathlib import Path
import time
from typing import Any, Callable

from transformers import AutoTokenizer

from experiments.lfm25_closed_track_eval import expanded_capability_rows
from experiments.lfm25_residual_stream_capability import LETTERS, task_scores
from heretic_nx.eval.capability import paired_bootstrap_interval
from heretic_nx.eval.comparison import validate_capability_pair
from heretic_nx.eval.gguf_runtime import (
    NativeRuntimeClient,
    RESTRICTED_GRAMMAR,
    attest_native_model,
    require_native_model_identity,
)
from heretic_nx.hashing import canonical_json, sha256_json

import gemma4_e2b_q8_eval as gemma


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "runs" / "gemma4-e2b-q8" / "capability"
BATCH_SIZE = 32
NONINFERIORITY_MARGIN = 0.03
ALPHA = 0.05
RESAMPLES = 10_000
SEED = 8314
SCHEMA_FAMILY = "gemma4-e2b"


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
    with ThreadPoolExecutor(max_workers=parallel) as pool:
        for start in range(completed, len(prompts), BATCH_SIZE):
            batch = prompts[start : start + BATCH_SIZE]
            started = time.perf_counter()
            produced = list(pool.map(worker, batch))
            checkpoint["seconds"] = float(checkpoint["seconds"]) + (
                time.perf_counter() - started
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
    runtime = attest_native_model(args.endpoint, artifact, expected_model=args.model)
    rows = expanded_capability_rows()
    rows_hash = sha256_json(rows)
    tokenizer = AutoTokenizer.from_pretrained(gemma.TOKENIZER_PATH, local_files_only=True)
    letter_ids = [tokenizer.encode(letter, add_special_tokens=False) for letter in LETTERS]
    if any(len(ids) != 1 for ids in letter_ids):
        raise RuntimeError(f"answer labels are not single tokens: {letter_ids}")
    answer_ids = [int(ids[0]) for ids in letter_ids]
    rendered = gemma.render(tokenizer, [str(row["prompt"]) for row in rows])
    prompts = [
        [int(token) for token in tokenizer.encode(value, add_special_tokens=False)]
        for value in rendered
    ]
    prompts_hash = sha256_json(prompts)
    expected = {
        "schema_version": f"{SCHEMA_FAMILY}-q8-capability-partial-v1",
        "label": args.label,
        "model": args.model,
        "runtime_model": runtime,
        "rows_sha256": rows_hash,
        "prompt_tokens_sha256": prompts_hash,
        "answer_token_ids": answer_ids,
        "grammar": RESTRICTED_GRAMMAR,
        "temperature": -1,
        "max_new_tokens": 1,
    }
    client = NativeRuntimeClient(args.endpoint)
    try:
        choices, seconds = resume_map(
            prompts=prompts,
            worker=client.restricted_choice,
            partial_path=RUN_DIR / f"{args.label}.partial.json",
            expected=expected,
            parallel=args.parallel,
        )
    finally:
        client.close()
    require_native_model_identity(args.endpoint, runtime, verify_artifact_hash=True)
    invalid = [value for value in choices if value not in LETTERS]
    if invalid:
        raise RuntimeError(f"restricted generation escaped labels: {invalid[:5]}")
    predictions = [LETTERS.index(choice) for choice in choices]
    correctness = [
        int(prediction == int(row["answer"]))
        for prediction, row in zip(predictions, rows)
    ]
    report = {
        "schema_version": f"{SCHEMA_FAMILY}-q8-capability-arm-v1",
        "label": args.label,
        "model": args.model,
        "artifact": {
            "filename": artifact.name,
            "sha256": runtime["artifact_sha256"],
            "size_bytes": runtime["artifact_size_bytes"],
        },
        "datasets": {
            "rows": len(rows),
            "rows_sha256": rows_hash,
            "tasks": sorted({str(row["task"]) for row in rows}),
        },
        "protocol": {
            "scoring": "restricted first-token A/B/C/D generation",
            "prompt_tokens_sha256": prompts_hash,
            "answer_token_ids": answer_ids,
            "grammar": RESTRICTED_GRAMMAR,
            "temperature": -1,
            "max_new_tokens": 1,
            "enable_thinking": False,
        },
        "runtime": runtime,
        "results": {
            "count": len(rows),
            "predictions": predictions,
            "correctness": correctness,
            "accuracy": sum(correctness) / len(correctness),
            "tasks": task_scores(rows, correctness),
            "seconds": seconds,
        },
        "interpretation_guard": (
            "This deterministic ARC-Challenge/HellaSwag/MMLU slice is broader "
            "than KL but is not a comprehensive capability benchmark."
        ),
    }
    report["evidence_sha256"] = sha256_json(report)
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
    base = json.loads((RUN_DIR / f"{args.base_label}.json").read_text())
    candidate = json.loads((RUN_DIR / f"{args.candidate_label}.json").read_text())
    rows = expanded_capability_rows()
    validate_capability_pair(base, candidate, rows)
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
    tasks = {}
    task_intervals = {}
    for task in sorted(base["results"]["tasks"]):
        indices = [i for i, row in enumerate(rows) if str(row["task"]) == task]
        task_intervals[task] = asdict(paired_bootstrap_interval(
            [base_correctness[i] for i in indices],
            [candidate_correctness[i] for i in indices],
            margin=NONINFERIORITY_MARGIN, alpha=ALPHA / len(base["results"]["tasks"]),
            resamples=RESAMPLES, seed=SEED,
        ))
        left = base["results"]["tasks"][task]
        right = candidate["results"]["tasks"][task]
        tasks[task] = {
            "count": left["count"],
            "base_accuracy": left["accuracy"],
            "candidate_accuracy": right["accuracy"],
            "difference": right["accuracy"] - left["accuracy"],
        }
    paired_counts = {
        "both_correct": sum(
            left == right == 1
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
            left == right == 0
            for left, right in zip(base_correctness, candidate_correctness)
        ),
    }
    report = {
        "schema_version": f"{SCHEMA_FAMILY}-q8-capability-comparison-v1",
        "base_label": args.base_label,
        "candidate_label": args.candidate_label,
        "rows": len(base_correctness),
        "rows_sha256": base["datasets"]["rows_sha256"],
        "protocol": base["protocol"],
        "artifacts": {"base": base["artifact"], "candidate": candidate["artifact"]},
        "results": {
            "base_accuracy": base["results"]["accuracy"],
            "candidate_accuracy": candidate["results"]["accuracy"],
            "candidate_minus_base": interval,
            "paired_counts": paired_counts,
            "tasks": tasks,
        },
        "passed_noninferiority": interval["noninferiority_passed"] and all(
            value["noninferiority_passed"] for value in task_intervals.values()),
        "simultaneous_task_intervals": task_intervals,
        "demonstrated_accuracy_improvement": interval["lower"] > 0.0,
        "interpretation_guard": (
            "Improvement requires the paired 95% interval above zero; "
            "noninferiority uses a predeclared three-point margin."
        ),
    }
    report["evidence_sha256"] = sha256_json(report)
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
        collect(args)
    else:
        compare(args)


if __name__ == "__main__":
    main()
