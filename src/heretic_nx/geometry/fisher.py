"""Low-rank empirical Fisher/Jacobian Gram factorization."""

from __future__ import annotations

import torch
from torch import Tensor


def fisher_factor_from_gradients(gradients: Tensor, rank: int) -> Tensor:
    """Return F with ``F F.T`` approximating ``E[g g.T]``."""

    values = gradients.float()
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("gradients must be a non-empty matrix")
    if rank < 1:
        raise ValueError("rank must be positive")
    _left, singular_values, right = torch.linalg.svd(values, full_matrices=False)
    selected = min(rank, singular_values.numel())
    return right[:selected].T * (singular_values[:selected] / values.shape[0] ** 0.5)
