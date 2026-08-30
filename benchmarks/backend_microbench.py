#!/usr/bin/env python3
"""Reproducible microbenchmarks for the rank-space optimizer and K codecs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import tempfile
import time

import numpy as np
import torch

from heretic_nx.edits.gguf_codecs import GGUFQuantizationCodecRegistry
from heretic_nx.edits.gguf_quant import _file_and_untouched_sha256
from heretic_nx.edits.matrix_opt import (
    _low_rank_frobenius_mean,
    _low_rank_spectral_norm,
)
from heretic_nx.hashing import sha256_file


def median_seconds(function, repeats: int) -> float:
    timings = []
    for _ in range(repeats):
        started = time.perf_counter()
        value = function()
        if isinstance(value, torch.Tensor):
            _ = float(value)
        timings.append(time.perf_counter() - started)
    return statistics.median(timings)


def legacy_two_pass_hashes(
    path: Path,
    intervals: tuple[tuple[int, int], ...],
    *,
    chunk_size: int = 8 * 1024 * 1024,
) -> tuple[str, str]:
    full = sha256_file(path)
    untouched = hashlib.sha256()
    file_size = path.stat().st_size
    with path.open("rb") as handle:
        cursor = 0
        for start, stop in (*intervals, (file_size, file_size)):
            handle.seek(cursor)
            remaining = start - cursor
            while remaining:
                chunk = handle.read(min(chunk_size, remaining))
                if not chunk:
                    raise RuntimeError("benchmark file ended unexpectedly")
                untouched.update(chunk)
                remaining -= len(chunk)
            cursor = stop
    return full, untouched.hexdigest()


def legacy_verification_cycle(
    path: Path,
    intervals: tuple[tuple[int, int], ...],
) -> tuple[str, str]:
    """Model the six integrity scans used before the one-pass merger."""

    source_sha256 = sha256_file(path)
    snapshot_sha256, untouched_before = legacy_two_pass_hashes(path, intervals)
    _final_sha256, untouched_after = legacy_two_pass_hashes(path, intervals)
    published_sha256 = sha256_file(path)
    if untouched_after != untouched_before:
        raise RuntimeError("legacy untouched digest changed")
    if source_sha256 != snapshot_sha256:
        raise RuntimeError("legacy source and snapshot digests differ")
    return snapshot_sha256, published_sha256


def combined_verification_cycle(
    path: Path,
    intervals: tuple[tuple[int, int], ...],
) -> tuple[str, str]:
    """Model the two integrity scans used by the optimized merger."""

    snapshot = _file_and_untouched_sha256(path, intervals)
    final = _file_and_untouched_sha256(path, intervals)
    if final.untouched_sha256 != snapshot.untouched_sha256:
        raise RuntimeError("combined untouched digest changed")
    return snapshot.sha256, final.sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimension", type=int, default=2048)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--compact-repeats", type=int, default=20)
    parser.add_argument("--dense-repeats", type=int, default=3)
    parser.add_argument("--codec-rows", type=int, default=1024)
    parser.add_argument("--codec-input-dim", type=int, default=4096)
    parser.add_argument("--io-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--io-repeats", type=int, default=3)
    parser.add_argument("--ggml-library")
    args = parser.parse_args()
    if min(
        args.dimension,
        args.rank,
        args.compact_repeats,
        args.dense_repeats,
        args.codec_rows,
        args.codec_input_dim,
        args.io_bytes,
        args.io_repeats,
    ) < 1:
        raise ValueError("benchmark dimensions and repeats must be positive")
    if args.rank > args.dimension:
        raise ValueError("rank cannot exceed dimension")

    generator = torch.Generator().manual_seed(251)
    a = torch.randn(args.dimension, args.rank, generator=generator)
    b = torch.randn(args.dimension, args.rank, generator=generator)
    for _ in range(3):
        _low_rank_frobenius_mean(a, b)
        _low_rank_spectral_norm(a, b)

    compact = median_seconds(
        lambda: _low_rank_frobenius_mean(a, b) + _low_rank_spectral_norm(a, b),
        args.compact_repeats,
    )
    dense = median_seconds(
        lambda: (a @ b.T).square().mean()
        + torch.linalg.matrix_norm(a @ b.T, ord=2),
        args.dense_repeats,
    )
    report: dict[str, object] = {
        "schema_version": "heretic-nx-backend-microbench-v1",
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(),
        },
        "rank_space": {
            "dimension": args.dimension,
            "rank": args.rank,
            "compact_median_seconds": compact,
            "dense_median_seconds": dense,
            "speedup": dense / compact,
        },
    }

    with tempfile.TemporaryDirectory(prefix="heretic-nx-io-bench-") as directory:
        io_path = Path(directory) / "synthetic.gguf"
        with io_path.open("wb") as handle:
            handle.truncate(args.io_bytes)
        intervals = (
            (args.io_bytes // 4, args.io_bytes // 4 + args.io_bytes // 16),
            (args.io_bytes // 2, args.io_bytes // 2 + args.io_bytes // 8),
        )
        expected = legacy_two_pass_hashes(io_path, intervals)
        combined = _file_and_untouched_sha256(io_path, intervals)
        if expected != (combined.sha256, combined.untouched_sha256):
            raise RuntimeError("combined I/O digests differ from the legacy baseline")
        legacy_seconds = median_seconds(
            lambda: legacy_verification_cycle(io_path, intervals),
            args.io_repeats,
        )
        combined_seconds = median_seconds(
            lambda: combined_verification_cycle(io_path, intervals),
            args.io_repeats,
        )
        report["integrity_io"] = {
            "file_bytes": args.io_bytes,
            "legacy_six_scan_median_seconds": legacy_seconds,
            "combined_two_scan_median_seconds": combined_seconds,
            "speedup": legacy_seconds / combined_seconds,
            "digests_match": True,
        }

    try:
        from gguf import GGMLQuantizationType

        registry = GGUFQuantizationCodecRegistry(ggml_library=args.ggml_library)
        values = np.ascontiguousarray(
            np.random.default_rng(257).normal(
                size=(args.codec_rows, args.codec_input_dim)
            ),
            dtype=np.float32,
        )
        codecs = []
        for name in ("Q4_K", "Q6_K"):
            qtype = getattr(GGMLQuantizationType, name)
            started = time.perf_counter()
            encoded = registry.quantize_rows(values, qtype)
            encode_seconds = time.perf_counter() - started
            started = time.perf_counter()
            decoded = registry.dequantize_rows(
                encoded,
                qtype,
                args.codec_input_dim,
            )
            decode_seconds = time.perf_counter() - started
            codecs.append(
                {
                    "quantization": name,
                    "input_bytes": values.nbytes,
                    "encoded_bytes": encoded.nbytes,
                    "encode_seconds": encode_seconds,
                    "decode_seconds": decode_seconds,
                    "finite": bool(np.isfinite(decoded).all()),
                }
            )
        report["k_quant_codecs"] = {
            "provenance": registry.provenance(),
            "measurements": codecs,
        }
    except (ImportError, RuntimeError) as error:
        report["k_quant_codecs"] = {"available": False, "error": str(error)}

    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
