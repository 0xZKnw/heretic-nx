#!/usr/bin/env python3
"""Reproducible microbenchmarks for the rank-space optimizer and K codecs."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time

import numpy as np
import torch

from heretic_nx.edits.gguf_codecs import GGUFQuantizationCodecRegistry
from heretic_nx.edits.matrix_opt import (
    _low_rank_frobenius_mean,
    _low_rank_spectral_norm,
)


def median_seconds(function, repeats: int) -> float:
    timings = []
    for _ in range(repeats):
        started = time.perf_counter()
        value = function()
        if isinstance(value, torch.Tensor):
            _ = float(value)
        timings.append(time.perf_counter() - started)
    return statistics.median(timings)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimension", type=int, default=2048)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--compact-repeats", type=int, default=20)
    parser.add_argument("--dense-repeats", type=int, default=3)
    parser.add_argument("--codec-rows", type=int, default=1024)
    parser.add_argument("--codec-input-dim", type=int, default=4096)
    parser.add_argument("--ggml-library")
    args = parser.parse_args()
    if min(
        args.dimension,
        args.rank,
        args.compact_repeats,
        args.dense_repeats,
        args.codec_rows,
        args.codec_input_dim,
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
