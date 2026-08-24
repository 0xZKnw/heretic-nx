"""Worst-case/CVaR objectives and hard scenario constraints."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


def smooth_max(losses: Tensor, temperature: float = 0.05) -> Tensor:
    if losses.ndim != 1 or losses.numel() == 0:
        raise ValueError("losses must be a non-empty vector")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return temperature * torch.logsumexp(losses / temperature, dim=0)


def cvar(losses: Tensor, alpha: float = 0.8) -> Tensor:
    if losses.ndim != 1 or losses.numel() == 0:
        raise ValueError("losses must be a non-empty vector")
    if not 0 <= alpha < 1:
        raise ValueError("alpha must be in [0, 1)")
    tail_count = max(1, int(torch.ceil(torch.tensor((1 - alpha) * losses.numel())).item()))
    return torch.topk(losses, tail_count).values.mean()


@dataclass(frozen=True)
class RobustFeasibility:
    feasible: bool
    worst_capability_drift: float
    worst_risk_drift: float
    violating_scenarios: tuple[int, ...]


def enforce_scenario_constraints(
    capability_drift: Tensor,
    risk_drift: Tensor,
    *,
    capability_maximum: float,
    risk_maximum: float,
) -> RobustFeasibility:
    if capability_drift.shape != risk_drift.shape or capability_drift.ndim != 1:
        raise ValueError("scenario drift vectors must be aligned")
    violations = torch.nonzero(
        (capability_drift > capability_maximum) | (risk_drift > risk_maximum)
    ).flatten()
    return RobustFeasibility(
        feasible=violations.numel() == 0,
        worst_capability_drift=float(capability_drift.max().item()),
        worst_risk_drift=float(risk_drift.max().item()),
        violating_scenarios=tuple(int(index) for index in violations.tolist()),
    )
