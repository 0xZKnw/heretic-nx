#!/usr/bin/env python3
"""Compare thin factorized LEACE with the former dense covariance fit."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time

import torch
from torch import Tensor

from heretic_nx.geometry.leace import fit_leace
from heretic_nx.geometry.principal_angles import orthonormal_basis


def _median_seconds(function, repeats: int) -> float:
    timings = []
    for _ in range(repeats):
        started = time.perf_counter()
        value = function()
        if isinstance(value, Tensor):
            _ = float(value.square().sum())
        timings.append(time.perf_counter() - started)
    return statistics.median(timings)


def _dense_fit(
    representations: Tensor,
    concepts: Tensor,
    *,
    tolerance: float = 1e-7,
) -> tuple[Tensor, Tensor, int]:
    """The pre-thin implementation, retained only as a benchmark oracle."""

    x = representations.float()
    z = concepts.float()
    mean_x = x.mean(dim=0)
    centered_x = x - mean_x
    centered_z = z - z.mean(dim=0)
    covariance = centered_x.T @ centered_x / x.shape[0]
    cross_covariance = centered_x.T @ centered_z / x.shape[0]
    eigenvalues, eigenvectors = torch.linalg.eigh((covariance + covariance.T) / 2)
    threshold = tolerance * eigenvalues.max().clamp_min(torch.finfo(x.dtype).eps)
    keep = eigenvalues > threshold
    support = eigenvectors[:, keep]
    roots = eigenvalues[keep].sqrt()
    covariance_sqrt = (support * roots) @ support.T
    whitening = (support / roots) @ support.T
    concept_basis = orthonormal_basis(
        whitening @ cross_covariance, tolerance=tolerance
    )
    if concept_basis.shape[1] == 0:
        projection = torch.eye(x.shape[1], dtype=x.dtype, device=x.device)
    else:
        projection = (
            torch.eye(x.shape[1], dtype=x.dtype, device=x.device)
            - covariance_sqrt @ concept_basis @ concept_basis.T @ whitening
        )
    bias = mean_x - projection @ mean_x
    return projection, bias, concept_basis.shape[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument("--dimension", type=int, default=2048)
    parser.add_argument("--categories", type=int, default=2)
    parser.add_argument("--fit-repeats", type=int, default=5)
    parser.add_argument("--factor-apply-repeats", type=int, default=100)
    parser.add_argument("--dense-apply-repeats", type=int, default=20)
    args = parser.parse_args()
    if min(
        args.rows,
        args.dimension,
        args.categories,
        args.fit_repeats,
        args.factor_apply_repeats,
        args.dense_apply_repeats,
    ) < 1:
        raise ValueError("benchmark arguments must be positive")
    if args.rows < 2:
        raise ValueError("rows must be at least two")

    generator = torch.Generator().manual_seed(359)
    labels = torch.randint(0, args.categories, (args.rows,), generator=generator)
    concepts = torch.nn.functional.one_hot(
        labels, num_classes=args.categories
    ).float()
    signal = concepts @ torch.randn(
        args.categories, args.dimension, generator=generator
    )
    values = signal + 0.3 * torch.randn(
        args.rows, args.dimension, generator=generator
    )

    for _ in range(3):
        fit_leace(values, concepts)
    thin_seconds = _median_seconds(
        lambda: fit_leace(values, concepts), args.fit_repeats
    )
    dense_projection, dense_bias, dense_concept_rank = _dense_fit(values, concepts)
    dense_seconds = _median_seconds(
        lambda: _dense_fit(values, concepts), args.fit_repeats
    )

    eraser = fit_leace(values, concepts)
    if eraser.erase_left is None or eraser.erase_right is None:
        raise RuntimeError("thin LEACE unexpectedly returned a dense eraser")
    thin_erased = eraser.apply(values)
    dense_erased = values @ dense_projection.T + dense_bias
    centered_concepts = concepts - concepts.mean(dim=0)
    erased_cross_covariance = (
        (thin_erased - thin_erased.mean(dim=0)).T
        @ centered_concepts
        / args.rows
    )
    relative_output_error = float(
        torch.linalg.vector_norm(thin_erased - dense_erased)
        / torch.linalg.vector_norm(dense_erased).clamp_min(
            torch.finfo(dense_erased.dtype).eps
        )
    )

    dense_delta_singular = torch.linalg.svdvals(
        torch.eye(args.dimension) - dense_projection
    )
    dense_delta_threshold = (
        1e-7
        * dense_delta_singular.max().clamp_min(
            torch.finfo(dense_delta_singular.dtype).eps
        )
    )
    dense_adapter_rank = int((dense_delta_singular > dense_delta_threshold).sum())

    for _ in range(3):
        eraser.apply(values)
        values @ dense_projection.T + dense_bias
    factor_apply_seconds = _median_seconds(
        lambda: eraser.apply(values), args.factor_apply_repeats
    )
    dense_apply_seconds = _median_seconds(
        lambda: values @ dense_projection.T + dense_bias,
        args.dense_apply_repeats,
    )

    element_size = values.element_size()
    factor_output_bytes = (
        eraser.erase_left.numel()
        + eraser.erase_right.numel()
        + eraser.bias.numel()
    ) * element_size
    dense_projection_bytes = args.dimension**2 * element_size
    # covariance, eigenvectors, covariance_sqrt, whitening, and projection are
    # simultaneously named by the old implementation.  Backend workspace is
    # deliberately excluded, making this a reproducible lower bound.
    dense_named_peak_lower_bound_bytes = 5 * dense_projection_bytes
    thin_named_peak_lower_bound_bytes = (
        2 * args.rows * args.dimension
        + args.rows * min(args.rows, args.dimension)
    ) * element_size

    report = {
        "schema_version": "heretic-nx-leace-thin-benchmark-v1",
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(),
        },
        "shape": [args.rows, args.dimension],
        "categories": args.categories,
        "thin_concept_rank": eraser.concept_rank,
        "dense_concept_rank": dense_concept_rank,
        "dense_adapter_numerical_rank": dense_adapter_rank,
        "fit": {
            "thin_median_seconds": thin_seconds,
            "dense_median_seconds": dense_seconds,
            "speedup": dense_seconds / thin_seconds,
        },
        "apply": {
            "factor_median_seconds": factor_apply_seconds,
            "dense_median_seconds": dense_apply_seconds,
            "speedup": dense_apply_seconds / factor_apply_seconds,
        },
        "memory": {
            "factorized_output_bytes": factor_output_bytes,
            "dense_projection_bytes": dense_projection_bytes,
            "output_compression_ratio": dense_projection_bytes / factor_output_bytes,
            "thin_named_peak_lower_bound_bytes": thin_named_peak_lower_bound_bytes,
            "dense_named_peak_lower_bound_bytes": dense_named_peak_lower_bound_bytes,
            "named_peak_lower_bound_ratio": (
                dense_named_peak_lower_bound_bytes
                / thin_named_peak_lower_bound_bytes
            ),
        },
        "accuracy": {
            "thin_vs_dense_calibration_relative_error": relative_output_error,
            "erased_cross_covariance_norm": float(erased_cross_covariance.norm()),
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
