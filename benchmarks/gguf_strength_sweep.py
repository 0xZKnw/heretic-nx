"""Reproducible exactness/performance benchmark for multi-strength GGUF sweeps.

The runner alternates sequential-first and sweep-first rounds, verifies every
output SHA-256 bit-for-bit, and reports medians for Q8_0/Q4_K in projector and
direct-right modes. Example:

    python benchmarks/gguf_strength_sweep.py --repeats 3 --warmups 1
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import statistics
import tempfile
import time

import numpy as np
from safetensors.numpy import save_file

from gguf import GGMLQuantizationType, GGUFWriter

from heretic_nx.edits.gguf_codecs import GGUFQuantizationCodecRegistry
from heretic_nx.edits.gguf_quant import (
    GGUFQuantizedAblationPlan,
    GGUFQuantizedTensorEdit,
    apply_quantized_gguf_ablation,
)
from heretic_nx.edits.gguf_sweep import (
    GGUFStrengthSweepCandidate,
    apply_quantized_gguf_strength_sweep,
)
from heretic_nx.hashing import sha256_file


def _build_fixture(
    root: Path,
    qtype: GGMLQuantizationType,
    mode: str,
    *,
    output_dim: int,
    input_dim: int,
    rank: int,
    strengths: tuple[float, ...],
    row_chunk_size: int,
    fast_search: bool,
) -> tuple[Path, Path, tuple[Path, ...]]:
    generator = np.random.default_rng(20260830)
    values = np.ascontiguousarray(
        generator.normal(scale=0.15, size=(output_dim, input_dim)),
        dtype=np.float32,
    )
    codec = GGUFQuantizationCodecRegistry()
    try:
        encoded = codec.quantize_rows(values, qtype)
    finally:
        codec.close()

    source = root / "source.gguf"
    writer = GGUFWriter(source, "llama")
    writer.add_name(f"heretic-nx-sweep-benchmark-{qtype.name}-{mode}")
    writer.add_tensor("blk.0.attn_output.weight", encoded, raw_dtype=qtype)
    writer.add_tensor("output_norm.weight", np.ones(output_dim, dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    factors = root / "factors.safetensors"
    left = np.ascontiguousarray(
        generator.normal(scale=0.03, size=(output_dim, rank)),
        dtype=np.float32,
    )
    if mode == "projector":
        left /= np.linalg.norm(left, axis=0, keepdims=True)
        save_file({"axis": left}, factors)
    else:
        right = np.ascontiguousarray(
            generator.normal(scale=0.03, size=(input_dim, rank)),
            dtype=np.float32,
        )
        save_file({"left": left, "right": right}, factors)

    plans = []
    for index, strength in enumerate(strengths):
        plan = root / f"plan-{index}.json"
        direct = mode == "direct"
        GGUFQuantizedAblationPlan(
            source_sha256=sha256_file(source),
            tensor_artifact_sha256=sha256_file(factors),
            row_chunk_size=row_chunk_size,
            verify_untouched_bytes=not fast_search,
            edits=(
                GGUFQuantizedTensorEdit(
                    tensor_name="blk.0.attn_output.weight",
                    expected_quantization=qtype.name,
                    a_key="left" if direct else "axis",
                    right_key="right" if direct else None,
                    strength=strength,
                    preserve_row_norms=not direct,
                    preserve_original_blocks=not direct,
                    quantization_multipliers=(
                        (1.0,) if direct else (0.75, 1.0, 1.25)
                    ),
                ),
            ),
        ).write(plan)
        plans.append(plan)
    return source, factors, tuple(plans)


def _sequential(
    source: Path,
    factors: Path,
    plans: tuple[Path, ...],
    output_dir: Path,
    *,
    fast_search: bool,
) -> tuple[float, tuple[Path, ...]]:
    outputs = tuple(output_dir / f"candidate-{index}.gguf" for index in range(len(plans)))
    started = time.perf_counter()
    for plan, output in zip(plans, outputs, strict=True):
        apply_quantized_gguf_ablation(
            source,
            output,
            plan,
            factors,
            fast_search=fast_search,
        )
    return time.perf_counter() - started, outputs


def _sweep(
    source: Path,
    factors: Path,
    plans: tuple[Path, ...],
    output_dir: Path,
    *,
    fast_search: bool,
) -> tuple[float, tuple[Path, ...]]:
    outputs = tuple(output_dir / f"candidate-{index}.gguf" for index in range(len(plans)))
    candidates = tuple(
        GGUFStrengthSweepCandidate(
            label=f"candidate-{index}",
            plan_path=plan,
            output_path=output,
        )
        for index, (plan, output) in enumerate(zip(plans, outputs, strict=True))
    )
    started = time.perf_counter()
    apply_quantized_gguf_strength_sweep(
        source,
        factors,
        candidates,
        fast_search=fast_search,
    )
    return time.perf_counter() - started, outputs


def _run_case(
    root: Path,
    qtype: GGMLQuantizationType,
    mode: str,
    args: argparse.Namespace,
) -> dict[str, object]:
    case_root = root / f"{qtype.name}-{mode}"
    case_root.mkdir()
    strengths = tuple(
        args.minimum_strength + index * args.strength_step
        for index in range(args.candidates)
    )
    source, factors, plans = _build_fixture(
        case_root,
        qtype,
        mode,
        output_dim=args.output_dim,
        input_dim=args.input_dim,
        rank=args.rank,
        strengths=strengths,
        row_chunk_size=args.row_chunk_size,
        fast_search=args.fast_search,
    )
    sequential_samples: list[float] = []
    sweep_samples: list[float] = []
    orders: list[str] = []
    for round_index in range(args.warmups + args.repeats):
        measured = round_index >= args.warmups
        sequential_dir = case_root / f"round-{round_index}-sequential"
        sweep_dir = case_root / f"round-{round_index}-sweep"
        sequential_dir.mkdir()
        sweep_dir.mkdir()
        if round_index % 2 == 0:
            order = "sequential,sweep"
            sequential_seconds, sequential_outputs = _sequential(
                source, factors, plans, sequential_dir, fast_search=args.fast_search
            )
            sweep_seconds, sweep_outputs = _sweep(
                source, factors, plans, sweep_dir, fast_search=args.fast_search
            )
        else:
            order = "sweep,sequential"
            sweep_seconds, sweep_outputs = _sweep(
                source, factors, plans, sweep_dir, fast_search=args.fast_search
            )
            sequential_seconds, sequential_outputs = _sequential(
                source, factors, plans, sequential_dir, fast_search=args.fast_search
            )
        sequential_hashes = tuple(sha256_file(path) for path in sequential_outputs)
        sweep_hashes = tuple(sha256_file(path) for path in sweep_outputs)
        if sequential_hashes != sweep_hashes:
            raise RuntimeError(f"exactness failure for {qtype.name}/{mode}")
        if measured:
            orders.append(order)
            sequential_samples.append(sequential_seconds)
            sweep_samples.append(sweep_seconds)

    sequential_median = statistics.median(sequential_samples)
    sweep_median = statistics.median(sweep_samples)
    return {
        "quantization": qtype.name,
        "mode": mode,
        "orders": orders,
        "sequential_seconds": sequential_samples,
        "sweep_seconds": sweep_samples,
        "sequential_median_seconds": sequential_median,
        "sweep_median_seconds": sweep_median,
        "speedup": sequential_median / sweep_median,
        "exact_sha256": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dim", type=int, default=2048)
    parser.add_argument("--input-dim", type=int, default=4096)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--row-chunk-size", type=int, default=128)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--minimum-strength", type=float, default=0.2)
    parser.add_argument("--strength-step", type=float, default=0.2)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--fast-search", action="store_true")
    args = parser.parse_args()
    if args.input_dim % 256:
        parser.error("--input-dim must be divisible by 256 for K-quants")
    if args.candidates < 2 or args.candidates > 32:
        parser.error("--candidates must be between 2 and 32")
    if args.repeats < 1 or args.warmups < 0:
        parser.error("--repeats must be positive and --warmups non-negative")

    with tempfile.TemporaryDirectory(prefix="heretic-nx-gguf-sweep-bench-") as name:
        root = Path(name)
        cases = [
            _run_case(root, qtype, mode, args)
            for qtype in (GGMLQuantizationType.Q8_0, GGMLQuantizationType.Q4_K)
            for mode in ("direct", "projector")
        ]
    print(
        json.dumps(
            {
                "schema_version": "heretic-nx-gguf-sweep-benchmark-v1",
                "platform": platform.platform(),
                "python": platform.python_version(),
                "cpu_count": os.cpu_count(),
                "parameters": vars(args),
                "cases": cases,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
