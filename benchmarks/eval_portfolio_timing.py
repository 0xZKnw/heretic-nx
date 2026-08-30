#!/usr/bin/env python3
"""Measure bounded candidate-wave scheduling overhead and concurrency.

The default worker is an I/O wait stand-in, not a model-performance claim.
Replace ``synthetic_worker`` with a real independent runtime call before using
the result as an inference speedup.  A single GPU should be benchmarked with
``--workers 1`` unless it genuinely supports independent concurrent slots.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from heretic_nx.eval.funnel import FunnelStage  # noqa: E402
from heretic_nx.eval.timing import (  # noqa: E402
    EvaluationWorkItem,
    execute_refusal_first_wave,
)


def median_wave_seconds(
    work: tuple[EvaluationWorkItem, ...],
    *,
    delay_seconds: float,
    workers: int,
    repeats: int,
) -> float:
    def synthetic_worker(_: EvaluationWorkItem) -> None:
        time.sleep(delay_seconds)

    samples: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        execute_refusal_first_wave(
            work,
            synthetic_worker,
            max_workers=workers,
        )
        samples.append(time.perf_counter() - started)
    return statistics.median(samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--delay-seconds", type=float, default=0.025)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.candidates < 1 or args.workers < 1 or args.repeats < 1:
        parser.error("candidates, workers, and repeats must be positive")
    if args.delay_seconds < 0.0:
        parser.error("delay-seconds must be non-negative")

    work = tuple(
        EvaluationWorkItem(f"{index + 1:064x}", FunnelStage.REFUSAL, 1)
        for index in range(args.candidates)
    )
    # Warm the thread-pool/import path before measuring both boundaries.
    median_wave_seconds(
        work,
        delay_seconds=0.0,
        workers=args.workers,
        repeats=1,
    )
    serial = median_wave_seconds(
        work,
        delay_seconds=args.delay_seconds,
        workers=1,
        repeats=args.repeats,
    )
    parallel = median_wave_seconds(
        work,
        delay_seconds=args.delay_seconds,
        workers=args.workers,
        repeats=args.repeats,
    )
    print(
        json.dumps(
            {
                "assumption": "independent I/O-bound candidate slots",
                "candidates": args.candidates,
                "delay_seconds_per_candidate": args.delay_seconds,
                "parallel_median_seconds": parallel,
                "repeats": args.repeats,
                "schema_version": "heretic-nx-eval-portfolio-benchmark-v1",
                "serial_median_seconds": serial,
                "speedup": serial / parallel if parallel > 0.0 else None,
                "workers": args.workers,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
