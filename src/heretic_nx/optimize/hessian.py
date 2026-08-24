"""Interaction-aware reduced Hessian estimation with shared random probes."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class ReducedHessian:
    diagonal: Tensor
    low_rank_factor: Tensor
    dense_estimate: Tensor
    probe_count: int

    def matrix(self) -> Tensor:
        return torch.diag(self.diagonal) + self.low_rank_factor @ self.low_rank_factor.T


def estimate_reduced_hessian(
    gradient_function: Callable[[Tensor], Tensor],
    dimension: int,
    *,
    probes: int = 12,
    epsilon: float = 0.03,
    residual_rank: int = 4,
    seed: int = 0,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> ReducedHessian:
    """Estimate H from central gradient HVPs along shared Rademacher probes."""

    if dimension < 1 or probes < 1 or epsilon <= 0:
        raise ValueError("dimension, probes, and epsilon must be positive")
    if residual_rank < 0:
        raise ValueError("residual_rank must be non-negative")
    target_device = torch.device(device)
    generator = torch.Generator(device=target_device).manual_seed(seed)
    directions = (
        torch.randint(0, 2, (probes, dimension), generator=generator, device=target_device)
        .to(dtype=dtype)
        * 2
        - 1
    )
    products = []
    for direction in directions:
        plus = gradient_function(epsilon * direction).to(device=target_device, dtype=dtype)
        minus = gradient_function(-epsilon * direction).to(device=target_device, dtype=dtype)
        if plus.shape != (dimension,) or minus.shape != (dimension,):
            raise ValueError("gradient_function must return one gradient per dimension")
        if not torch.isfinite(plus).all() or not torch.isfinite(minus).all():
            raise ValueError("gradient_function returned non-finite values")
        products.append((plus - minus) / (2 * epsilon))
    hvps = torch.stack(products)
    # directions @ H.T = hvps; solve least squares and symmetrize.
    dense = torch.linalg.lstsq(directions, hvps).solution.T
    dense = (dense + dense.T) / 2
    diagonal = torch.diag(dense).clamp_min(0)
    residual = (dense - torch.diag(diagonal) + (dense - torch.diag(diagonal)).T) / 2
    eigenvalues, eigenvectors = torch.linalg.eigh(residual)
    positive = torch.nonzero(eigenvalues > torch.finfo(dtype).eps).flatten()
    if positive.numel():
        selected = positive[-min(residual_rank, positive.numel()) :]
        factor = eigenvectors[:, selected] * eigenvalues[selected].sqrt()
    else:
        factor = dense.new_empty((dimension, 0))
    return ReducedHessian(diagonal, factor, dense, probes)
