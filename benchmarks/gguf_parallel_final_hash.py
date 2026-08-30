"""Benchmark exact concurrent final hashing for GGUF strength sweeps.

Pass two or more immutable GGUF candidates. Repeating the same path models
copy-on-write candidates that still share most physical blocks. Example:

    python benchmarks/gguf_parallel_final_hash.py model.gguf model.gguf \
        --warmups 1 --repeats 3
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import platform
import statistics
import time

from heretic_nx.edits.gguf_quant import _file_and_untouched_sha256


def _hash(path: Path, *, verify_untouched: bool) -> object:
    return _file_and_untouched_sha256(
        path,
        (),
        capture_untouched=verify_untouched,
    )


def _serial(paths: tuple[Path, ...], *, verify_untouched: bool) -> tuple[object, ...]:
    return tuple(_hash(path, verify_untouched=verify_untouched) for path in paths)


def _parallel(
    paths: tuple[Path, ...],
    *,
    workers: int,
    verify_untouched: bool,
) -> tuple[object, ...]:
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return tuple(
            executor.map(
                lambda path: _hash(path, verify_untouched=verify_untouched),
                paths,
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--verify-untouched", action="store_true")
    args = parser.parse_args()
    if len(args.paths) < 2:
        parser.error("at least two candidate paths are required")
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.warmups < 0 or args.repeats < 1:
        parser.error("--warmups must be non-negative and --repeats positive")
    paths = tuple(path.expanduser().resolve(strict=True) for path in args.paths)
    if any(not path.is_file() for path in paths):
        parser.error("every path must be a regular file")
    workers = min(args.workers, len(paths), max(1, os.cpu_count() or 1))

    serial_samples: list[float] = []
    parallel_samples: list[float] = []
    orders: list[str] = []
    for round_index in range(args.warmups + args.repeats):
        measured = round_index >= args.warmups
        if round_index % 2 == 0:
            order = "serial,parallel"
            started = time.perf_counter()
            serial = _serial(paths, verify_untouched=args.verify_untouched)
            serial_seconds = time.perf_counter() - started
            started = time.perf_counter()
            parallel = _parallel(
                paths,
                workers=workers,
                verify_untouched=args.verify_untouched,
            )
            parallel_seconds = time.perf_counter() - started
        else:
            order = "parallel,serial"
            started = time.perf_counter()
            parallel = _parallel(
                paths,
                workers=workers,
                verify_untouched=args.verify_untouched,
            )
            parallel_seconds = time.perf_counter() - started
            started = time.perf_counter()
            serial = _serial(paths, verify_untouched=args.verify_untouched)
            serial_seconds = time.perf_counter() - started
        if parallel != serial:
            raise RuntimeError("parallel final hashes differ from serial hashes")
        if measured:
            orders.append(order)
            serial_samples.append(serial_seconds)
            parallel_samples.append(parallel_seconds)

    serial_median = statistics.median(serial_samples)
    parallel_median = statistics.median(parallel_samples)
    print(
        json.dumps(
            {
                "schema_version": "heretic-nx-gguf-final-hash-benchmark-v1",
                "platform": platform.platform(),
                "python": platform.python_version(),
                "paths": [str(path) for path in paths],
                "sizes_bytes": [path.stat().st_size for path in paths],
                "workers": workers,
                "verify_untouched": args.verify_untouched,
                "orders": orders,
                "serial_seconds": serial_samples,
                "parallel_seconds": parallel_samples,
                "serial_median_seconds": serial_median,
                "parallel_median_seconds": parallel_median,
                "speedup": serial_median / parallel_median,
                "exact_snapshots": True,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
