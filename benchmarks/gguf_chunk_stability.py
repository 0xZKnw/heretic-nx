#!/usr/bin/env python3
"""Benchmark GGUF payload stability across streaming chunk sizes."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
import types

import numpy as np

from heretic_nx.edits.gguf_codecs import GGUFQuantizationCodecRegistry
from heretic_nx.edits.gguf_quant import _ResolvedEdit, _edit_tensor_payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=1024)
    parser.add_argument("--input-dim", type=int, default=4096)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--banks", type=int, default=1)
    parser.add_argument("--qtype", default="Q4_K")
    parser.add_argument("--chunks", type=int, nargs="+", default=[64, 128, 257])
    parser.add_argument(
        "--multipliers",
        type=float,
        nargs="+",
        default=[0.75, 1.0, 1.25],
    )
    parser.add_argument("--requantize-all", action="store_true")
    parser.add_argument("--direct-right", action="store_true")
    parser.add_argument(
        "--arithmetic-mode",
        choices=["chunk-stable-v1", "legacy-plan-v2"],
        default="chunk-stable-v1",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--ggml-library")
    args = parser.parse_args()
    if min(
        args.rows,
        args.input_dim,
        args.rank,
        args.banks,
        args.repeats,
        *args.chunks,
    ) < 1:
        raise ValueError("benchmark dimensions and controls must be positive")
    if min(args.multipliers) <= 0:
        raise ValueError("quantization multipliers must be positive")

    try:
        from gguf import GGMLQuantizationType
    except ImportError as error:
        raise RuntimeError("benchmark requires the 'gguf' extra") from error
    try:
        qtype = getattr(GGMLQuantizationType, args.qtype)
    except AttributeError as error:
        raise ValueError(f"unknown quantization type: {args.qtype}") from error

    generator = np.random.default_rng(367)
    values = np.ascontiguousarray(
        generator.normal(
            scale=0.08,
            size=(args.banks, args.rows, args.input_dim),
        ),
        dtype=np.float32,
    )
    factor = np.ascontiguousarray(
        generator.normal(size=(args.rows, args.rank)),
        dtype=np.float32,
    )
    factor /= np.linalg.norm(factor, axis=0, keepdims=True)
    right = np.ascontiguousarray(
        generator.normal(size=(args.input_dim, args.rank)),
        dtype=np.float32,
    )
    right /= np.linalg.norm(right, axis=0, keepdims=True)
    edit = _ResolvedEdit(
        tensor_name="benchmark.weight",
        expected_quantization=args.qtype,
        a_key="axis",
        b_key=None,
        right_key="right" if args.direct_right else None,
        strength=0.7,
        preserve_row_norms=not args.direct_right,
        preserve_original_blocks=not args.requantize_all,
        quantization_multipliers=tuple(args.multipliers),
        minimum_block_improvement=0.0,
        require_payload_change=True,
        minimum_delta_cosine=None,
        maximum_delta_relative_error=None,
        maximum_row_norm_relative_error=None,
    )

    with GGUFQuantizationCodecRegistry(
        ggml_library=args.ggml_library,
    ) as codec:
        original = codec.quantize_rows(
            values.reshape(-1, args.input_dim),
            qtype,
        ).reshape(args.banks, args.rows, -1)
        measurements = []
        for chunk_size in args.chunks:
            timings = []
            report = None
            payload = None
            for _ in range(args.repeats):
                payload = original.copy()
                tensor = types.SimpleNamespace(
                    tensor_type=qtype,
                    shape=np.array([args.input_dim, args.rows, args.banks]),
                    data=payload,
                    name=edit.tensor_name,
                )
                started = time.perf_counter()
                report = _edit_tensor_payload(
                    tensor,
                    edit,
                    {"axis": factor, "right": right},
                    codec,
                    row_chunk_size=chunk_size,
                    arithmetic_mode=args.arithmetic_mode,
                )
                timings.append(time.perf_counter() - started)
            assert report is not None and payload is not None
            measurements.append(
                {
                    "row_chunk_size": chunk_size,
                    "median_seconds": statistics.median(timings),
                    "payload_sha256": report["after_payload_sha256"],
                    "changed_bytes": report["quantization_metrics"]["changed_bytes"],
                    "tracked_chunk_array_bytes_lower_bound": report[
                        "quantization_metrics"
                    ]["tracked_chunk_array_bytes_lower_bound"],
                    "metrics_sha256": hashlib.sha256(
                        json.dumps(
                            report["quantization_metrics"],
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode("utf-8")
                    ).hexdigest(),
                }
            )
        codec_provenance = codec.provenance()

    hashes = {row["payload_sha256"] for row in measurements}
    print(
        json.dumps(
            {
                "schema_version": "heretic-nx-gguf-chunk-stability-benchmark-v2",
                "shape": [args.banks, args.rows, args.input_dim],
                "rank": args.rank,
                "quantization": args.qtype,
                "preserve_original_blocks": not args.requantize_all,
                "quantization_multipliers": args.multipliers,
                "direct_right": args.direct_right,
                "arithmetic_mode": args.arithmetic_mode,
                "repeats": args.repeats,
                "codec": codec_provenance,
                "payload_identical_across_chunks": len(hashes) == 1,
                "measurements": measurements,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
