"""Measure one-pass KL integrity validation on realistic vocabulary rows."""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
from statistics import median
import tempfile
import time

import numpy as np

from heretic_nx.eval.kl_integrity import (
    LOG_PROBABILITY_MASS_ATOL,
    load_completed_progress,
    load_completed_raw_logits,
    validate_log_probability_matrix,
)
from heretic_nx.hashing import sha256_file


SCHEMA = "benchmark-first-token-raw-logits-v1"


def _log_probability_mass(row: np.ndarray) -> float:
    maximum = float(np.max(row))
    return maximum + math.log(float(np.exp(row - maximum).sum()))


def _legacy_load_raw(
    data_path: Path,
    progress_path: Path,
    *,
    rows: int,
    vocab_size: int,
) -> tuple[np.memmap, dict[str, object]]:
    """Pre-optimization two-pass hash plus double-normalization reference."""

    progress = load_completed_progress(
        progress_path,
        schema_version=SCHEMA,
        count=rows,
        vocab_size=vocab_size,
    )
    digest = sha256_file(data_path)
    if digest != progress["data_sha256"]:
        raise RuntimeError("benchmark raw hash mismatch")
    matrix = np.memmap(
        data_path,
        mode="r",
        dtype=np.float32,
        shape=(rows, vocab_size),
    )
    for row_index in range(rows):
        row = np.asarray(matrix[row_index], dtype=np.float64)
        if not np.isfinite(row).all():
            raise RuntimeError("benchmark non-finite row")
        if float(np.max(row)) == float(np.min(row)):
            raise RuntimeError("benchmark degenerate row")
        normalized = row - _log_probability_mass(row)
        log_mass = _log_probability_mass(normalized)
        if abs(log_mass) > LOG_PROBABILITY_MASS_ATOL:
            raise RuntimeError("benchmark invalid normalized row")
    return matrix, progress


def _legacy_validate_log_probabilities(matrix: np.ndarray, path: Path) -> None:
    for row_index in range(matrix.shape[0]):
        row = np.asarray(matrix[row_index], dtype=np.float64)
        if not np.isfinite(row).all():
            raise RuntimeError("benchmark non-finite row")
        if abs(_log_probability_mass(row)) > LOG_PROBABILITY_MASS_ATOL:
            raise RuntimeError("benchmark invalid log-probability row")


def _timed(call, repeats: int):
    samples: list[float] = []
    result = call()
    for _ in range(repeats):
        started = time.perf_counter()
        result = call()
        samples.append(time.perf_counter() - started)
    return median(samples), result


def _write_progress(
    path: Path,
    *,
    data_path: Path,
    rows: int,
    vocab_size: int,
) -> None:
    artifact_sha256 = "2" * 64
    path.write_text(
        json.dumps(
            {
                "schema_version": SCHEMA,
                "label": "candidate",
                "model": "candidate.gguf",
                "prompt_tokens_sha256": "1" * 64,
                "artifact_sha256": artifact_sha256,
                "data_sha256": sha256_file(data_path),
                "runtime_model": {
                    "endpoint": "http://127.0.0.1:1236",
                    "model_alias": "candidate.gguf",
                    "model_ftype": "Q8_0",
                    "model_path": "/models/candidate.gguf",
                    "artifact_sha256": artifact_sha256,
                    "artifact_size_bytes": 1024,
                    "build_info": "benchmark",
                },
                "vocab_size": vocab_size,
                "count": rows,
                "completed": rows,
                "seconds": 1.0,
            }
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=104)
    parser.add_argument("--vocab-size", type=int, default=128_000)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    if min(args.rows, args.vocab_size, args.repeats) < 1:
        parser.error("all numeric arguments must be positive")
    if args.vocab_size < 2:
        parser.error("--vocab-size must be at least two")

    with tempfile.TemporaryDirectory(prefix="heretic-kl-integrity-") as temp:
        root = Path(temp)
        raw_path = root / "candidate.raw.bin"
        array_path = root / "candidate.npy"
        progress_path = root / "candidate.raw.progress.json"
        generator = np.random.default_rng(20260830)
        matrix = generator.standard_normal(
            (args.rows, args.vocab_size), dtype=np.float32
        )
        for row in matrix:
            row64 = np.asarray(row, dtype=np.float64)
            row -= np.float32(_log_probability_mass(row64))
        matrix.tofile(raw_path)
        np.save(array_path, matrix, allow_pickle=False)
        del matrix
        gc.collect()
        _write_progress(
            progress_path,
            data_path=raw_path,
            rows=args.rows,
            vocab_size=args.vocab_size,
        )
        log_probabilities = np.load(array_path, mmap_mode="r", allow_pickle=False)

        legacy_raw = lambda: _legacy_load_raw(
            raw_path,
            progress_path,
            rows=args.rows,
            vocab_size=args.vocab_size,
        )
        optimized_raw = lambda: load_completed_raw_logits(
            raw_path,
            progress_path,
            schema_version=SCHEMA,
            count=args.rows,
            vocab_size=args.vocab_size,
        )
        legacy_validation = lambda: _legacy_validate_log_probabilities(
            log_probabilities, array_path
        )
        optimized_validation = lambda: validate_log_probability_matrix(
            log_probabilities,
            path=array_path,
            shape=(args.rows, args.vocab_size),
        )

        legacy_raw_seconds, legacy_result = _timed(legacy_raw, args.repeats)
        optimized_raw_seconds, optimized_result = _timed(
            optimized_raw, args.repeats
        )
        legacy_validation_seconds, _ = _timed(
            legacy_validation, args.repeats
        )
        optimized_validation_seconds, _ = _timed(
            optimized_validation, args.repeats
        )

        row_f32_bytes = args.vocab_size * np.dtype(np.float32).itemsize
        row_f64_bytes = args.vocab_size * np.dtype(np.float64).itemsize
        legacy_peak_workspace = max(8 * 1024 * 1024, 4 * row_f64_bytes)
        optimized_peak_workspace = row_f32_bytes + row_f64_bytes
        print(
            json.dumps(
                {
                    "shape": [args.rows, args.vocab_size],
                    "artifact_bytes": raw_path.stat().st_size,
                    "raw_legacy_seconds_median": legacy_raw_seconds,
                    "raw_optimized_seconds_median": optimized_raw_seconds,
                    "raw_speedup": legacy_raw_seconds / optimized_raw_seconds,
                    "raw_full_data_passes": {"legacy": 2, "optimized": 1},
                    "legacy_peak_workspace_bytes_estimate": (
                        legacy_peak_workspace
                    ),
                    "optimized_peak_workspace_bytes": optimized_peak_workspace,
                    "workspace_reduction": (
                        legacy_peak_workspace / optimized_peak_workspace
                    ),
                    "logprob_legacy_seconds_median": (
                        legacy_validation_seconds
                    ),
                    "logprob_optimized_seconds_median": (
                        optimized_validation_seconds
                    ),
                    "logprob_speedup": (
                        legacy_validation_seconds
                        / optimized_validation_seconds
                    ),
                    "sha256_equal": (
                        legacy_result[1]["data_sha256"]
                        == optimized_result[1]["data_sha256"]
                    ),
                    "mapped_values_equal": bool(
                        np.array_equal(legacy_result[0], optimized_result[0])
                    ),
                },
                indent=2,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
