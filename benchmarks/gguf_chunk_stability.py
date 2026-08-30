#!/usr/bin/env python3
"""Benchmark GGUF payload stability across streaming chunk sizes."""

from __future__ import annotations

import argparse
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
    parser.add_argument("--qtype", default="Q4_K")
    parser.add_argument("--chunks", type=int, nargs="+", default=[64, 128, 257])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--ggml-library")
    args = parser.parse_args()
    if min(
        args.rows,
        args.input_dim,
        args.rank,
        args.repeats,
        *args.chunks,
    ) < 1:
        raise ValueError("benchmark dimensions and controls must be positive")

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
        generator.normal(scale=0.08, size=(args.rows, args.input_dim)),
        dtype=np.float32,
    )
    factor = np.ascontiguousarray(
        generator.normal(size=(args.rows, args.rank)),
        dtype=np.float32,
    )
    factor /= np.linalg.norm(factor, axis=0, keepdims=True)
    edit = _ResolvedEdit(
        tensor_name="benchmark.weight",
        expected_quantization=args.qtype,
        a_key="axis",
        b_key=None,
        right_key=None,
        strength=0.7,
        preserve_row_norms=True,
        preserve_original_blocks=True,
        quantization_multipliers=(0.75, 1.0, 1.25),
        minimum_block_improvement=0.0,
        require_payload_change=True,
        minimum_delta_cosine=None,
        maximum_delta_relative_error=None,
        maximum_row_norm_relative_error=None,
    )

    with GGUFQuantizationCodecRegistry(
        ggml_library=args.ggml_library,
    ) as codec:
        original = codec.quantize_rows(values, qtype)
        measurements = []
        for chunk_size in args.chunks:
            timings = []
            report = None
            payload = None
            for _ in range(args.repeats):
                payload = original.copy()
                tensor = types.SimpleNamespace(
                    tensor_type=qtype,
                    shape=np.array([args.input_dim, args.rows]),
                    data=payload,
                    name=edit.tensor_name,
                )
                started = time.perf_counter()
                report = _edit_tensor_payload(
                    tensor,
                    edit,
                    {"axis": factor},
                    codec,
                    row_chunk_size=chunk_size,
                )
                timings.append(time.perf_counter() - started)
            assert report is not None and payload is not None
            measurements.append(
                {
                    "row_chunk_size": chunk_size,
                    "median_seconds": statistics.median(timings),
                    "payload_sha256": report["after_payload_sha256"],
                    "changed_bytes": report["quantization_metrics"]["changed_bytes"],
                }
            )
        codec_provenance = codec.provenance()

    hashes = {row["payload_sha256"] for row in measurements}
    print(
        json.dumps(
            {
                "schema_version": "heretic-nx-gguf-chunk-stability-benchmark-v1",
                "shape": [args.rows, args.input_dim],
                "rank": args.rank,
                "quantization": args.qtype,
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
