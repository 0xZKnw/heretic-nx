#!/usr/bin/env python3
"""Compare deterministic exact sample-Gram PCA with the former randomized fit."""

from __future__ import annotations

import argparse
import json
import statistics
import time

import torch

from heretic_nx.geometry.pca import exact_principal_components


def median_seconds(function, repeats: int) -> float:
    values = []
    for _ in range(repeats):
        started = time.perf_counter()
        function()
        values.append(time.perf_counter() - started)
    return statistics.median(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=400)
    parser.add_argument("--dimension", type=int, default=2048)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if min(args.rows, args.dimension, args.rank, args.repeats) < 1:
        raise ValueError("benchmark arguments must be positive")

    generator = torch.Generator().manual_seed(349)
    samples = torch.randn(args.rows, args.dimension, generator=generator)
    centered = samples - samples.mean(dim=0, keepdim=True)

    def randomized():
        torch.manual_seed(349)
        return torch.pca_lowrank(
            centered,
            q=min(args.rank + 4, args.rows, args.dimension),
            center=False,
            niter=3,
        )

    exact = exact_principal_components(
        samples,
        maximum_rank=args.rank,
    )
    _u, randomized_singular, randomized_vectors = randomized()
    randomized_basis = randomized_vectors[:, : args.rank]
    randomized_energy = float(
        randomized_singular[: args.rank].square().sum()
        / centered.square().sum()
    )
    subspace_cosine_minimum = float(
        torch.linalg.svdvals(exact.basis.T @ randomized_basis).min()
    )
    report = {
        "schema_version": "heretic-nx-pca-microbench-v1",
        "shape": [args.rows, args.dimension],
        "rank": args.rank,
        "exact_median_seconds": median_seconds(
            lambda: exact_principal_components(samples, maximum_rank=args.rank),
            args.repeats,
        ),
        "randomized_median_seconds": median_seconds(randomized, args.repeats),
        "exact_retained_energy_fraction": exact.retained_energy_fraction,
        "randomized_retained_energy_fraction": randomized_energy,
        "randomized_vs_exact_minimum_subspace_cosine": subspace_cosine_minimum,
        "exact_effective_rank": exact.effective_rank,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
