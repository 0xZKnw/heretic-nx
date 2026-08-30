"""Measure durable per-verdict SQLite overhead versus explicit batching."""

from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
from pathlib import Path

from heretic_nx.eval.cache import JudgeCache
from heretic_nx.eval.cascade import JudgeCascade, JudgeInput, JudgeVerdict


def _entries(count: int) -> list[tuple[str, dict[str, object]]]:
    return [
        (
            f"key-{index:06d}",
            {
                "label": "compliance",
                "confidence": 0.95,
                "level": "J1",
                "rationale": "benchmark",
            },
        )
        for index in range(count)
    ]


def _median_seconds(run, repeats: int) -> float:
    return statistics.median(run() for _ in range(repeats))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=104)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    entries = _entries(args.count)
    judge_inputs = [
        JudgeInput(f"prompt-{index:06d}", "ambiguous benchmark response")
        for index in range(args.count)
    ]

    def j1(_prompt: str, _response: str, _rubric: str) -> JudgeVerdict:
        return JudgeVerdict("compliance", 0.95, "J1", rationale="benchmark")

    with tempfile.TemporaryDirectory(prefix="hnx-judge-cache-") as root:
        root_path = Path(root)
        run_index = 0

        def fresh_path(kind: str) -> Path:
            nonlocal run_index
            run_index += 1
            return root_path / f"{kind}-{run_index}.sqlite"

        def single_writes() -> float:
            with JudgeCache(fresh_path("single")) as cache:
                started = time.perf_counter()
                for key, payload in entries:
                    cache.put(key, payload)
                return time.perf_counter() - started

        def transactional_writes() -> float:
            with JudgeCache(fresh_path("transaction")) as cache:
                started = time.perf_counter()
                with cache.transaction():
                    for key, payload in entries:
                        cache.put(key, payload)
                return time.perf_counter() - started

        def batch_writes() -> float:
            with JudgeCache(fresh_path("batch")) as cache:
                started = time.perf_counter()
                cache.put_many(entries)
                return time.perf_counter() - started

        read_path = fresh_path("reads")
        with JudgeCache(read_path) as cache:
            cache.put_many(entries)

        def single_reads() -> float:
            with JudgeCache(read_path) as cache:
                started = time.perf_counter()
                for key, _payload in entries:
                    cache.get(key)
                return time.perf_counter() - started

        def batch_reads() -> float:
            with JudgeCache(read_path) as cache:
                started = time.perf_counter()
                cache.get_many(key for key, _payload in entries)
                return time.perf_counter() - started

        def cascade_single() -> float:
            with JudgeCache(fresh_path("cascade-single")) as cache:
                cascade = JudgeCascade(rubric="benchmark-v1", j1=j1, cache=cache)
                started = time.perf_counter()
                for item in judge_inputs:
                    cascade.judge(item.prompt, item.response)
                return time.perf_counter() - started

        def cascade_batch() -> float:
            with JudgeCache(fresh_path("cascade-batch")) as cache:
                cascade = JudgeCascade(rubric="benchmark-v1", j1=j1, cache=cache)
                started = time.perf_counter()
                cascade.judge_many(judge_inputs)
                return time.perf_counter() - started

        # Run once before timing so imports and SQLite initialization are warm.
        single_writes()
        transactional_writes()
        batch_writes()
        single_reads()
        batch_reads()
        cascade_single()
        cascade_batch()

        single_write_s = _median_seconds(single_writes, args.repeats)
        transaction_write_s = _median_seconds(transactional_writes, args.repeats)
        batch_write_s = _median_seconds(batch_writes, args.repeats)
        single_read_s = _median_seconds(single_reads, args.repeats)
        batch_read_s = _median_seconds(batch_reads, args.repeats)
        cascade_single_s = _median_seconds(cascade_single, args.repeats)
        cascade_batch_s = _median_seconds(cascade_batch, args.repeats)

    print(
        json.dumps(
            {
                "count": args.count,
                "repeats": args.repeats,
                "single_durable_write_s": single_write_s,
                "explicit_transaction_write_s": transaction_write_s,
                "put_many_write_s": batch_write_s,
                "write_speedup_transaction": single_write_s / transaction_write_s,
                "write_speedup_put_many": single_write_s / batch_write_s,
                "single_read_s": single_read_s,
                "get_many_read_s": batch_read_s,
                "read_speedup_get_many": single_read_s / batch_read_s,
                "cascade_single_s": cascade_single_s,
                "cascade_judge_many_s": cascade_batch_s,
                "cascade_speedup_judge_many": cascade_single_s / cascade_batch_s,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
