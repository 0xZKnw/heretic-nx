#!/usr/bin/env python3
"""Reproducible sparse-application and optimizer-stop microbenchmarks."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time

import torch

from heretic_nx.edits.activation_op import ActivationOperator, metric_projector_operator
from heretic_nx.edits.matrix_opt import fit_low_rank_matrix_operator
from heretic_nx.geometry.metric import LowRankMetric


def _median_seconds(function, repeats: int) -> float:
    timings = []
    for _ in range(repeats):
        started = time.perf_counter()
        result = function()
        if isinstance(result, torch.Tensor):
            _ = float(result.reshape(-1)[0])
        timings.append(time.perf_counter() - started)
    return statistics.median(timings)


def _legacy_sparse_apply(operator: ActivationOperator, hidden: torch.Tensor) -> torch.Tensor:
    delta = (hidden @ operator.b.to(hidden)) @ operator.a.to(hidden).T
    mask = torch.zeros(operator.dimension, dtype=hidden.dtype, device=hidden.device)
    assert operator.sparse_index is not None
    mask[operator.sparse_index.to(hidden.device)] = 1
    return hidden - operator.beta * delta * mask


def _legacy_metric_projector(
    basis: torch.Tensor,
    metric: LowRankMetric,
    *,
    beta: float,
    ridge: float = 1e-6,
) -> torch.Tensor:
    gram = basis.T @ metric.apply(basis)
    inverse_gram = torch.linalg.inv(
        gram + ridge * torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
    )
    return metric.apply(basis) @ inverse_gram


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimension", type=int, default=4096)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--sparse-coordinates", type=int, default=64)
    parser.add_argument("--metric-rank", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--constructor-repeats", type=int, default=30)
    parser.add_argument("--optimizer-steps", type=int, default=200)
    parser.add_argument("--optimizer-repeats", type=int, default=5)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--torch-threads", type=int, default=1)
    args = parser.parse_args()
    if min(
        args.dimension,
        args.rank,
        args.tokens,
        args.sparse_coordinates,
        args.metric_rank,
        args.repeats,
        args.constructor_repeats,
        args.optimizer_steps,
        args.optimizer_repeats,
        args.patience,
        args.torch_threads,
    ) < 1:
        raise ValueError("benchmark dimensions and controls must be positive")
    if args.rank > args.dimension or args.sparse_coordinates > args.dimension:
        raise ValueError("rank and sparse coordinates cannot exceed dimension")
    torch.set_num_threads(args.torch_threads)

    generator = torch.Generator().manual_seed(317)
    operator = ActivationOperator(
        a=torch.randn(args.dimension, args.rank, generator=generator),
        b=torch.randn(args.dimension, args.rank, generator=generator),
        beta=0.73,
        sparse_index=torch.randperm(args.dimension, generator=generator)[
            : args.sparse_coordinates
        ],
    ).bounded()
    hidden = torch.randn(args.tokens, args.dimension, generator=generator)
    for _ in range(3):
        _legacy_sparse_apply(operator, hidden)
        operator.apply(hidden)
    expected = _legacy_sparse_apply(operator, hidden)
    actual = operator.apply(hidden)
    if not torch.allclose(actual, expected, atol=2e-6, rtol=2e-6):
        raise RuntimeError("indexed sparse output differs from mask baseline")
    maximum_absolute_error = float((actual - expected).abs().max())
    legacy_seconds = _median_seconds(
        lambda: _legacy_sparse_apply(operator, hidden), args.repeats
    )
    indexed_seconds = _median_seconds(lambda: operator.apply(hidden), args.repeats)

    metric = LowRankMetric.from_factors(
        args.dimension,
        covariance_factor=torch.randn(
            args.dimension, args.metric_rank, generator=generator
        ),
    )
    basis = torch.linalg.qr(
        torch.randn(args.dimension, args.rank, generator=generator), mode="reduced"
    )[0]
    legacy_b = _legacy_metric_projector(basis, metric, beta=0.73)
    optimized_b = metric_projector_operator(basis, metric, beta=0.73).b
    if not torch.equal(legacy_b, optimized_b):
        raise RuntimeError("cached metric application changed projector factors")
    legacy_constructor_seconds = _median_seconds(
        lambda: _legacy_metric_projector(basis, metric, beta=0.73),
        args.constructor_repeats,
    )
    optimized_constructor_seconds = _median_seconds(
        lambda: metric_projector_operator(basis, metric, beta=0.73).b,
        args.constructor_repeats,
    )

    optimizer_generator = torch.Generator().manual_seed(19)
    protected = torch.randn(64, 32, generator=optimizer_generator)
    target = torch.randn(64, 32, generator=optimizer_generator)
    target[:, :3] += torch.tensor([3.0, 2.0, 1.0])
    common = {
        "rank": 4,
        "beta": 0.8,
        "steps": args.optimizer_steps,
        "seed": 7,
    }
    fit_low_rank_matrix_operator(target, protected, rank=4, beta=0.8, steps=2, seed=7)
    full_timings = []
    stopped_timings = []
    full = None
    stopped = None
    for _ in range(args.optimizer_repeats):
        started = time.perf_counter()
        full = fit_low_rank_matrix_operator(target, protected, **common)
        full_timings.append(time.perf_counter() - started)
        started = time.perf_counter()
        stopped = fit_low_rank_matrix_operator(
            target,
            protected,
            **common,
            early_stopping_patience=args.patience,
        )
        stopped_timings.append(time.perf_counter() - started)
    assert full is not None and stopped is not None
    full_seconds = statistics.median(full_timings)
    stopped_seconds = statistics.median(stopped_timings)

    print(
        json.dumps(
            {
                "schema_version": "heretic-nx-activation-operator-benchmark-v1",
                "environment": {
                    "platform": platform.platform(),
                    "python": platform.python_version(),
                    "torch": torch.__version__,
                    "torch_threads": torch.get_num_threads(),
                },
                "sparse_apply": {
                    "dimension": args.dimension,
                    "rank": args.rank,
                    "tokens": args.tokens,
                    "selected_coordinates": args.sparse_coordinates,
                    "legacy_mask_median_seconds": legacy_seconds,
                    "indexed_median_seconds": indexed_seconds,
                    "speedup": legacy_seconds / indexed_seconds,
                    "maximum_absolute_error": maximum_absolute_error,
                },
                "optimizer_early_stop": {
                    "maximum_steps": args.optimizer_steps,
                    "full_steps": full.steps,
                    "stopped_steps": stopped.steps,
                    "full_median_seconds": full_seconds,
                    "stopped_median_seconds": stopped_seconds,
                    "speedup": full_seconds / stopped_seconds,
                    "full_best_loss": full.final_loss,
                    "full_terminal_loss": full.terminal_loss,
                    "best_over_terminal_improvement": (
                        full.terminal_loss / full.final_loss
                        if full.terminal_loss is not None
                        else None
                    ),
                    "best_step": full.best_step,
                    "stopped_best_loss": stopped.final_loss,
                    "same_best_loss": full.final_loss == stopped.final_loss,
                },
                "metric_projector_construction": {
                    "dimension": args.dimension,
                    "basis_rank": args.rank,
                    "metric_rank": args.metric_rank,
                    "legacy_two_apply_median_seconds": legacy_constructor_seconds,
                    "cached_one_apply_median_seconds": optimized_constructor_seconds,
                    "speedup": (
                        legacy_constructor_seconds / optimized_constructor_seconds
                    ),
                    "bit_exact": True,
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
