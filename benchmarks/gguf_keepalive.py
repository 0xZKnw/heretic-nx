#!/usr/bin/env python3
"""Benchmark persistent HTTP connections without changing inference requests."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import statistics
import time
from typing import Callable

from heretic_nx.eval.gguf_runtime import (
    attest_native_model,
    native_completion,
    native_restricted_choice,
    NativeRuntimeClient,
    require_native_model_identity,
)
from heretic_nx.hashing import sha256_json


def load_prompt_tokens(path: Path, *, limit: int | None) -> list[list[int]]:
    prompts: list[list[int]] = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line:
            raise ValueError(f"empty prompt-token row at line {line_number}")
        try:
            prompts.append([int(value) for value in line.split()])
        except ValueError as error:
            raise ValueError(
                f"invalid integer prompt token at line {line_number}"
            ) from error
        if limit is not None and len(prompts) == limit:
            break
    if not prompts:
        raise ValueError("prompt-token file contains no rows")
    return prompts


def checkpointed_map(
    prompts: list[list[int]],
    *,
    worker: Callable[[list[int]], str],
    parallel: int,
    checkpoint_size: int,
) -> list[str]:
    values: list[str] = []
    with ThreadPoolExecutor(max_workers=parallel) as pool:
        for start in range(0, len(prompts), checkpoint_size):
            values.extend(
                pool.map(worker, prompts[start : start + checkpoint_size])
            )
    return values


def timed(function: Callable[[], list[str]]) -> tuple[list[str], float]:
    started = time.perf_counter()
    value = function()
    return value, time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--model")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8080")
    parser.add_argument("--prompt-tokens", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("restricted", "generation"),
        default="restricted",
    )
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--parallel", type=int, default=8)
    parser.add_argument("--checkpoint-size", type=int, default=32)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    for name in ("max_tokens", "parallel", "checkpoint_size", "repetitions"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("limit must be positive")
    if args.mode == "restricted" and args.max_tokens != 1:
        raise ValueError("restricted mode requires --max-tokens 1")

    prompts = load_prompt_tokens(args.prompt_tokens, limit=args.limit)
    identity = attest_native_model(
        args.endpoint,
        args.artifact,
        expected_model=args.model,
    )
    client = NativeRuntimeClient(args.endpoint)
    if args.mode == "restricted":
        legacy_worker = lambda prompt: native_restricted_choice(args.endpoint, prompt)
        keepalive_worker = client.restricted_choice
    else:
        legacy_worker = lambda prompt: native_completion(
            args.endpoint,
            prompt,
            max_tokens=args.max_tokens,
        )
        keepalive_worker = lambda prompt: client.completion(
            prompt,
            max_tokens=args.max_tokens,
        )

    def legacy() -> list[str]:
        return checkpointed_map(
            prompts,
            worker=legacy_worker,
            parallel=args.parallel,
            checkpoint_size=args.checkpoint_size,
        )

    def keepalive() -> list[str]:
        return checkpointed_map(
            prompts,
            worker=keepalive_worker,
            parallel=args.parallel,
            checkpoint_size=args.checkpoint_size,
        )

    reference = legacy()
    if keepalive() != reference:
        raise RuntimeError("persistent-connection warm-up changed at least one output")
    timings = {"new_connection": [], "persistent_connection": []}
    for repetition in range(args.repetitions):
        order = (
            ("new_connection", legacy),
            ("persistent_connection", keepalive),
        )
        if repetition % 2:
            order = tuple(reversed(order))
        for label, function in order:
            values, seconds = timed(function)
            if values != reference:
                raise RuntimeError(f"{label} changed output at repetition {repetition}")
            timings[label].append(seconds)

    require_native_model_identity(
        args.endpoint,
        identity,
        verify_artifact_hash=True,
    )
    legacy_median = statistics.median(timings["new_connection"])
    keepalive_median = statistics.median(timings["persistent_connection"])
    report = {
        "schema_version": "heretic-nx-gguf-keepalive-benchmark-v1",
        "artifact": identity,
        "protocol": {
            "mode": args.mode,
            "prompt_count": len(prompts),
            "prompt_tokens_sha256": sha256_json(prompts),
            "max_tokens": args.max_tokens,
            "parallel": args.parallel,
            "checkpoint_size": args.checkpoint_size,
            "repetitions": args.repetitions,
        },
        "exact_outputs": True,
        "timings_seconds": timings,
        "median_seconds": {
            "new_connection": legacy_median,
            "persistent_connection": keepalive_median,
        },
        "rows_per_second": {
            "new_connection": len(prompts) / legacy_median,
            "persistent_connection": len(prompts) / keepalive_median,
        },
        "speedup": legacy_median / keepalive_median,
        "interpretation_guard": (
            "This isolates HTTP connection reuse while preserving one native "
            "request per prompt, prompt order, checkpoint barriers, and concurrency. "
            "Re-run it for each server build, hardware, model, and prompt shape."
        ),
    }
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
