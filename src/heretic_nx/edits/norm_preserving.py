"""Row-norm-preserving static weight interventions."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor

from .activation_op import ActivationOperator


@torch.no_grad()
def norm_preserving_weight_edit(
    weight: Tensor,
    operator: ActivationOperator,
    *,
    strength: float,
    collapse_tolerance: float = 1e-7,
) -> Tensor:
    """Apply ``I-strength*AB.T`` and exactly restore each output-row norm.

    The intervention is computed in FP32 and returned in the input dtype.  A
    non-zero row that collapses to a numerically undefined direction fails
    closed instead of silently becoming a zero row.
    """

    if weight.ndim != 2 or operator.dimension != weight.shape[0]:
        raise ValueError("operator must live in the weight output space")
    if not math.isfinite(strength) or strength < 0:
        raise ValueError("strength must be finite and non-negative")
    if not math.isfinite(collapse_tolerance) or collapse_tolerance <= 0:
        raise ValueError("collapse_tolerance must be finite and positive")
    if not torch.isfinite(weight).all():
        raise ValueError("weight must be finite")

    original = weight.float()
    row_norms = torch.linalg.vector_norm(original, dim=1, keepdim=True)
    normalized = F.normalize(original, p=2, dim=1)
    a = operator.a.to(device=weight.device, dtype=torch.float32)
    b = operator.b.to(device=weight.device, dtype=torch.float32)
    effective_strength = strength * operator.beta
    edited = normalized - effective_strength * a @ (b.T @ normalized)
    edited_norms = torch.linalg.vector_norm(edited, dim=1, keepdim=True)
    nonzero = row_norms.squeeze(1) > collapse_tolerance
    collapsed = nonzero & (edited_norms.squeeze(1) <= collapse_tolerance)
    if bool(collapsed.any()):
        raise RuntimeError("norm-preserving intervention collapsed a non-zero row")
    restored = F.normalize(edited, p=2, dim=1) * row_norms
    if not torch.isfinite(restored).all():
        raise RuntimeError("norm-preserving intervention became non-finite")
    return restored.to(dtype=weight.dtype)
