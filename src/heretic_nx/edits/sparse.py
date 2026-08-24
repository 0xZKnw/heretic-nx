"""Deterministic Atomic-Unit scoring and selection."""

from __future__ import annotations

import torch
from torch import Tensor


def atomic_unit_scores(
    target_separation: Tensor,
    scenario_stability: Tensor,
    protected_sensitivity: Tensor,
    *,
    epsilon: float = 1e-8,
) -> Tensor:
    if not (
        target_separation.ndim == scenario_stability.ndim == protected_sensitivity.ndim == 1
        and target_separation.shape == scenario_stability.shape == protected_sensitivity.shape
    ):
        raise ValueError("atomic-unit signals must be aligned vectors")
    return target_separation.abs() * scenario_stability.clamp_min(0) / (
        protected_sensitivity.abs() + epsilon
    )


def select_atomic_units(scores: Tensor, count: int) -> Tensor:
    if scores.ndim != 1:
        raise ValueError("scores must be one-dimensional")
    if count < 1 or count > scores.numel():
        raise ValueError("count must be within the score dimension")
    # stable=True gives deterministic tie-breaking by original coordinate.
    return torch.argsort(scores, descending=True, stable=True)[:count].sort().values
