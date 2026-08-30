#!/usr/bin/env python3
"""Benchmark deterministic row-parallel native GGUF quantization."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import time

import numpy as np

from heretic_nx.edits.gguf_codecs import (
    GGUFQuantizationCodecRegistry,
    QUANT_LAYOUTS,
)


def median_quantization(
    codec: GGUFQuantizationCodecRegistry,
    values: np.ndarray,
    qtype: object,
    *,
    repeats: int,
) -> tuple[float, np.ndarray]:
    codec.quantize_rows(values, qtype)
    timings = []
    result = None
    for _ in range(repeats):
        started = time.perf_counter()
        result = codec.quantize_rows(values, qtype)
        timings.append(time.perf_counter() - started)
    assert result is not None
    return statistics.median(timings), result


def median_dequantization(
    codec: GGUFQuantizationCodecRegistry,
    encoded: np.ndarray,
    qtype: object,
    *,
    input_dim: int,
    repeats: int,
) -> tuple[float, np.ndarray]:
    codec.dequantize_rows(encoded, qtype, input_dim)
    timings = []
    result = None
    for _ in range(repeats):
        started = time.perf_counter()
        result = codec.dequantize_rows(encoded, qtype, input_dim)
        timings.append(time.perf_counter() - started)
    assert result is not None
    return statistics.median(timings), result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=1024)
    parser.add_argument("--input-dim", type=int, default=4096)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--threads", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument(
        "--qtypes",
        nargs="+",
        default=["Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K"],
    )
    parser.add_argument("--ggml-library")
    args = parser.parse_args()
    if min(args.rows, args.input_dim, args.repeats, *args.threads) < 1:
        raise ValueError("rows, dimensions, repeats, and threads must be positive")
    unknown = sorted(set(args.qtypes) - set(QUANT_LAYOUTS))
    if unknown:
        raise ValueError(f"unsupported qtypes: {unknown}")
    incompatible = [
        name
        for name in args.qtypes
        if args.input_dim % QUANT_LAYOUTS[name].block_size
    ]
    if incompatible:
        raise ValueError(
            f"input dimension {args.input_dim} is incompatible with {incompatible}"
        )

    try:
        from gguf import GGMLQuantizationType
    except ImportError as error:
        raise RuntimeError("benchmark requires the 'gguf' extra") from error

    values = np.ascontiguousarray(
        np.random.default_rng(263).normal(size=(args.rows, args.input_dim)),
        dtype=np.float32,
    )
    report: dict[str, object] = {
        "schema_version": "heretic-nx-gguf-codec-parallel-benchmark-v2",
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "workload": {
            "rows": args.rows,
            "input_dim": args.input_dim,
            "input_bytes": values.nbytes,
            "repeats": args.repeats,
        },
        "measurements": [],
    }
    measurements: list[dict[str, object]] = []
    with GGUFQuantizationCodecRegistry(
        ggml_library=args.ggml_library,
        quantization_threads=1,
    ) as serial:
        for name in args.qtypes:
            qtype = getattr(GGMLQuantizationType, name)
            serial_seconds, reference = median_quantization(
                serial, values, qtype, repeats=args.repeats
            )
            serial_dequant_seconds, reference_dequantized = median_dequantization(
                serial,
                reference,
                qtype,
                input_dim=args.input_dim,
                repeats=args.repeats,
            )
            rows: list[dict[str, object]] = []
            for threads in args.threads:
                with GGUFQuantizationCodecRegistry(
                    ggml_library=args.ggml_library,
                    quantization_threads=threads,
                    parallel_min_elements=1,
                ) as parallel:
                    parallel_seconds, candidate = median_quantization(
                        parallel, values, qtype, repeats=args.repeats
                    )
                    parallel_dequant_seconds, candidate_dequantized = (
                        median_dequantization(
                            parallel,
                            candidate,
                            qtype,
                            input_dim=args.input_dim,
                            repeats=args.repeats,
                        )
                    )
                rows.append(
                    {
                        "threads": threads,
                        "quantization": {
                            "median_seconds": parallel_seconds,
                            "speedup": serial_seconds / parallel_seconds,
                            "bit_identical": bool(np.array_equal(reference, candidate)),
                        },
                        "dequantization": {
                            "median_seconds": parallel_dequant_seconds,
                            "speedup": (
                                serial_dequant_seconds / parallel_dequant_seconds
                            ),
                            "bit_identical": bool(
                                np.array_equal(
                                    reference_dequantized,
                                    candidate_dequantized,
                                )
                            ),
                        },
                    }
                )
            measurements.append(
                {
                    "quantization": name,
                    "encoded_bytes": reference.nbytes,
                    "output_sha256": hashlib.sha256(memoryview(reference)).hexdigest(),
                    "serial_quantization_median_seconds": serial_seconds,
                    "serial_dequantization_median_seconds": serial_dequant_seconds,
                    "parallel": rows,
                }
            )
        report["codec"] = serial.provenance()
    report["measurements"] = measurements
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
