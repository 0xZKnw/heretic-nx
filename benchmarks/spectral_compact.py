#!/usr/bin/env python3
"""Compare the compact signed-contrast eigensolver with the dense baseline."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time

import torch

from heretic_nx.edits.spectral import fit_signed_spectral_operator


def _median_seconds(function, repeats: int) -> float:
    timings = []
    for _ in range(repeats):
        started = time.perf_counter()
        value = function()
        _ = float(value.contrast_eigenvalues.abs().sum())
        timings.append(time.perf_counter() - started)
    return statistics.median(timings)


def _dense_fit(
    target: torch.Tensor,
    protected: torch.Tensor,
    *,
    rank: int,
    beta: float,
):
    target_normalized = target / torch.linalg.vector_norm(target)
    protected_normalized = protected / torch.linalg.vector_norm(protected)
    contrast = target_normalized @ target_normalized.T - (
        protected_normalized @ protected_normalized.T
    )
    contrast = (contrast + contrast.T) / 2
    eigenvalues, eigenvectors = torch.linalg.eigh(contrast)
    order = torch.argsort(eigenvalues.abs(), descending=True, stable=True)
    selected = order[eigenvalues[order].abs() > 1e-7][:rank]
    values = eigenvalues[selected]
    coefficients = values / values.abs().max().clamp_min(torch.finfo(values.dtype).eps)
    basis = eigenvectors[:, selected]
    from heretic_nx.edits.activation_op import ActivationOperator
    from heretic_nx.edits.spectral import SignedSpectralEdit

    return SignedSpectralEdit(
        ActivationOperator(a=basis * coefficients, b=basis, beta=beta),
        basis,
        coefficients,
        values,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimension", type=int, default=2048)
    parser.add_argument("--factor-rank", type=int, default=8)
    parser.add_argument("--selected-rank", type=int, default=8)
    parser.add_argument("--compact-repeats", type=int, default=20)
    parser.add_argument("--dense-repeats", type=int, default=3)
    args = parser.parse_args()
    if min(
        args.dimension,
        args.factor_rank,
        args.selected_rank,
        args.compact_repeats,
        args.dense_repeats,
    ) < 1:
        raise ValueError("dimensions, ranks, and repeats must be positive")
    if 2 * args.factor_rank > args.dimension:
        raise ValueError("the benchmark requires 2 * factor-rank <= dimension")

    generator = torch.Generator().manual_seed(337)
    target = torch.randn(args.dimension, args.factor_rank, generator=generator)
    protected = torch.randn(args.dimension, args.factor_rank, generator=generator)
    compact_function = lambda: fit_signed_spectral_operator(
        target,
        protected,
        rank=args.selected_rank,
        beta=0.5,
        positive_only=False,
    )
    dense_function = lambda: _dense_fit(
        target,
        protected,
        rank=args.selected_rank,
        beta=0.5,
    )
    for _ in range(3):
        compact_function()
    dense_function()

    compact = _median_seconds(compact_function, args.compact_repeats)
    dense = _median_seconds(dense_function, args.dense_repeats)
    compact_result = compact_function()
    dense_result = dense_function()
    overlap = torch.linalg.svdvals(dense_result.basis.T @ compact_result.basis)
    eigenvalue_error = (
        dense_result.contrast_eigenvalues - compact_result.contrast_eigenvalues
    ).abs().max()
    report = {
        "schema_version": "heretic-nx-spectral-compact-benchmark-v1",
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(),
        },
        "dimension": args.dimension,
        "factor_rank": args.factor_rank,
        "selected_rank": args.selected_rank,
        "compact_core_maximum_dimension": 2 * args.factor_rank,
        "dense_ambient_matrix_bytes": args.dimension**2 * target.element_size(),
        "compact_joined_factor_bytes": (
            2 * args.dimension * args.factor_rank * target.element_size()
        ),
        "compact_median_seconds": compact,
        "dense_median_seconds": dense,
        "speedup": dense / compact,
        "minimum_subspace_cosine": float(overlap.min()),
        "maximum_eigenvalue_error": float(eigenvalue_error),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
