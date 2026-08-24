"""Box-constrained KKT water-filling for edit intensities."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class KKTResult:
    beta: Tensor
    multiplier: float
    spent_budget: float


def solve_kkt(
    gain: Tensor,
    curvature: Tensor,
    cost: Tensor,
    budget: float,
    beta_max: float | Tensor,
    tolerance: float = 1e-7,
    max_iterations: int = 100,
) -> KKTResult:
    gain, curvature, cost = torch.broadcast_tensors(gain, curvature, cost)
    if torch.any(curvature <= 0) or torch.any(cost < 0):
        raise ValueError("curvature must be positive and cost must be non-negative")
    if budget < 0:
        raise ValueError("budget must be non-negative")
    cap = torch.as_tensor(beta_max, device=gain.device, dtype=gain.dtype)

    def beta(multiplier: float) -> Tensor:
        return torch.clamp((gain - multiplier * cost) / curvature, min=0)\
            .minimum(cap)

    unconstrained = beta(0.0)
    if float(torch.dot(cost.flatten(), unconstrained.flatten()).item()) <= budget:
        spent = float(torch.dot(cost.flatten(), unconstrained.flatten()).item())
        return KKTResult(unconstrained, 0.0, spent)

    low, high = 0.0, 1.0
    while float(torch.dot(cost.flatten(), beta(high).flatten()).item()) > budget:
        high *= 2.0
        if high > 1e20:
            raise RuntimeError("failed to bracket the KKT multiplier")

    for _ in range(max_iterations):
        middle = (low + high) / 2.0
        if float(torch.dot(cost.flatten(), beta(middle).flatten()).item()) > budget:
            low = middle
        else:
            high = middle
        if high - low <= tolerance:
            break

    solution = beta(high)
    spent = float(torch.dot(cost.flatten(), solution.flatten()).item())
    return KKTResult(solution, high, spent)
