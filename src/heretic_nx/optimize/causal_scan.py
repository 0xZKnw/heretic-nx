"""Cheap symmetric causal estimates used before autoregressive evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class SymmetricEstimate:
    gradient: Tensor
    curvature: Tensor


def symmetric_estimate(
    minus: Tensor,
    baseline: Tensor,
    plus: Tensor,
    epsilon: float,
) -> SymmetricEstimate:
    """Estimate first and second derivatives from ``-eps, 0, +eps`` probes."""

    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if minus.shape != baseline.shape or plus.shape != baseline.shape:
        raise ValueError("all observations must have the same shape")
    gradient = (plus - minus) / (2.0 * epsilon)
    curvature = (plus - 2.0 * baseline + minus) / (epsilon * epsilon)
    return SymmetricEstimate(gradient=gradient, curvature=curvature)


def dual_representation_loss(
    benign_candidate: Tensor,
    benign_base_teacher: Tensor,
    unsafe_candidate: Tensor,
    unsafe_safe_teacher: Tensor,
    *,
    unsafe_weight: float = 1.0,
) -> Tensor:
    """Match the intact model on benign inputs and safe behavior on unsafe ones."""

    if unsafe_weight < 0:
        raise ValueError("unsafe_weight must be non-negative")
    benign_loss = torch.mean((benign_candidate - benign_base_teacher) ** 2)
    unsafe_loss = torch.mean((unsafe_candidate - unsafe_safe_teacher) ** 2)
    return benign_loss + unsafe_weight * unsafe_loss
