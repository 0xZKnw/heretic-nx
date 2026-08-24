"""Robust architecture-independent contrastive direction estimation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class ContrastiveAxis:
    axis: Tensor
    fold_cosine_minimum: float
    fold_cosine_mean: float
    safe_mean_cosine: float
    folds: int


def _remove_axis(value: Tensor, axis: Tensor, tolerance: float) -> Tensor:
    norm = torch.linalg.vector_norm(axis)
    if float(norm) <= tolerance:
        return value
    unit = axis / norm
    return value - unit * torch.dot(unit, value)


def fit_contrastive_axis(
    safe: Tensor,
    target: Tensor,
    *,
    folds: int = 3,
    remove_safe_mean: bool = True,
    tolerance: float = 1e-7,
) -> ContrastiveAxis:
    """Estimate a stable target-minus-safe axis from interleaved folds.

    Fold directions are sign-aligned to the global contrast before averaging.
    Optionally removing the safe mean prevents the edit from directly erasing
    the dominant benign offset while leaving the estimator model-agnostic.
    """

    safe_values = safe.float()
    target_values = target.float()
    if (
        safe_values.ndim != 2
        or target_values.ndim != 2
        or safe_values.shape != target_values.shape
    ):
        raise ValueError("safe and target states must be aligned matrices")
    if safe_values.shape[0] < 2 * folds or folds < 2:
        raise ValueError("each contrastive fold requires at least two examples")
    if not torch.isfinite(safe_values).all() or not torch.isfinite(target_values).all():
        raise ValueError("contrastive states must be finite")
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")

    safe_mean = safe_values.mean(dim=0)
    global_axis = target_values.mean(dim=0) - safe_mean
    if remove_safe_mean:
        global_axis = _remove_axis(global_axis, safe_mean, tolerance)
    global_norm = torch.linalg.vector_norm(global_axis)
    if float(global_norm) <= tolerance:
        raise RuntimeError("global contrastive direction collapsed")
    global_unit = global_axis / global_norm

    fold_units = []
    for fold in range(folds):
        direction = (
            target_values[fold::folds].mean(dim=0)
            - safe_values[fold::folds].mean(dim=0)
        )
        if remove_safe_mean:
            direction = _remove_axis(direction, safe_mean, tolerance)
        norm = torch.linalg.vector_norm(direction)
        if float(norm) <= tolerance:
            raise RuntimeError(f"contrastive fold {fold} collapsed")
        unit = direction / norm
        if float(torch.dot(unit, global_unit)) < 0:
            unit = -unit
        fold_units.append(unit)

    stacked = torch.stack(fold_units)
    consensus = stacked.mean(dim=0)
    consensus_norm = torch.linalg.vector_norm(consensus)
    if float(consensus_norm) <= tolerance:
        raise RuntimeError("contrastive consensus collapsed")
    axis = F.normalize(consensus, dim=0)
    cosine = stacked @ axis
    safe_mean_cosine = 0.0
    safe_mean_norm = torch.linalg.vector_norm(safe_mean)
    if float(safe_mean_norm) > tolerance:
        safe_mean_cosine = float(torch.dot(axis, safe_mean / safe_mean_norm))
    return ContrastiveAxis(
        axis=axis,
        fold_cosine_minimum=float(cosine.min()),
        fold_cosine_mean=float(cosine.mean()),
        safe_mean_cosine=safe_mean_cosine,
        folds=folds,
    )
