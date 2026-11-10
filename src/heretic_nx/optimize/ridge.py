"""Benign-penalized ridge detector fitted with matrix-free conjugate gradients."""

from __future__ import annotations

import math

import torch


def conjugate_gradient(
    harmful: torch.Tensor,
    safe: torch.Tensor,
    exact: torch.Tensor,
    *,
    safe_lambda: float,
    steps: int = 40,
    tolerance: float = 1e-5,
    ridge_fraction: float = 1e-4,
) -> tuple[torch.Tensor, int, float]:
    if (
        harmful.ndim != 2
        or safe.ndim != 2
        or harmful.shape[1] != safe.shape[1]
        or exact.shape != (harmful.shape[1],)
    ):
        raise ValueError("ridge detector dimensions disagree")
    if min(len(harmful), len(safe)) < 2 or not all(
        torch.isfinite(value).all() for value in (harmful, safe, exact)
    ):
        raise ValueError("ridge detector states must be finite nonempty matrices")
    if (
        isinstance(steps, bool)
        or not isinstance(steps, int)
        or steps < 1
        or not math.isfinite(safe_lambda)
        or safe_lambda < 0
        or not math.isfinite(tolerance)
        or tolerance <= 0
        or not math.isfinite(ridge_fraction)
        or ridge_fraction <= 0
    ):
        raise ValueError("invalid ridge solver parameters")
    harmful_count = float(len(harmful))
    safe_count = float(len(safe))
    ridge = float(
        ridge_fraction * (harmful.square().mean() + safe_lambda * safe.square().mean())
    )

    def covariance(
        values: torch.Tensor, vector: torch.Tensor, count: float
    ) -> torch.Tensor:
        return values.T @ (values @ vector) / count

    def system(vector: torch.Tensor) -> torch.Tensor:
        return (
            covariance(harmful, vector, harmful_count)
            + safe_lambda * covariance(safe, vector, safe_count)
            + ridge * vector
        )

    right = covariance(harmful, exact, harmful_count)
    solution = torch.zeros_like(right)
    residual = right.clone()
    direction = residual.clone()
    initial_norm = torch.linalg.vector_norm(residual).clamp_min(1e-12)
    residual_square = torch.dot(residual, residual)
    residual_ratio = 1.0
    iterations = 0
    for iteration in range(1, steps + 1):
        product = system(direction)
        alpha = residual_square / torch.dot(direction, product).clamp_min(1e-20)
        solution = solution + alpha * direction
        new_residual = residual - alpha * product
        residual_ratio = float(torch.linalg.vector_norm(new_residual) / initial_norm)
        iterations = iteration
        if residual_ratio <= tolerance:
            break
        new_square = torch.dot(new_residual, new_residual)
        direction = new_residual + (new_square / residual_square) * direction
        residual = new_residual
        residual_square = new_square
    return solution.detach().cpu(), iterations, residual_ratio
