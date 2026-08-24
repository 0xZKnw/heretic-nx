"""Closed-form linear concept erasure baseline."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .principal_angles import orthonormal_basis


@dataclass(frozen=True)
class LeaceEraser:
    projection: Tensor
    bias: Tensor
    concept_rank: int

    def apply(self, values: Tensor) -> Tensor:
        return values @ self.projection.T + self.bias


def fit_leace(
    representations: Tensor,
    concepts: Tensor,
    *,
    tolerance: float = 1e-7,
) -> LeaceEraser:
    """Fit LEACE from centered covariance and cross-covariance matrices."""

    x = representations.float()
    z = concepts.float()
    if x.ndim != 2 or z.ndim != 2 or x.shape[0] != z.shape[0]:
        raise ValueError("representations and concepts must be aligned matrices")
    if x.shape[0] < 2:
        raise ValueError("at least two samples are required")
    mean_x = x.mean(dim=0)
    centered_x = x - mean_x
    centered_z = z - z.mean(dim=0)
    covariance = centered_x.T @ centered_x / x.shape[0]
    cross_covariance = centered_x.T @ centered_z / x.shape[0]
    eigenvalues, eigenvectors = torch.linalg.eigh((covariance + covariance.T) / 2)
    threshold = tolerance * eigenvalues.max().clamp_min(torch.finfo(x.dtype).eps)
    keep = eigenvalues > threshold
    support = eigenvectors[:, keep]
    roots = eigenvalues[keep].sqrt()
    covariance_sqrt = (support * roots) @ support.T
    whitening = (support / roots) @ support.T
    concept_basis = orthonormal_basis(whitening @ cross_covariance, tolerance=tolerance)
    if concept_basis.shape[1] == 0:
        projection = torch.eye(x.shape[1], dtype=x.dtype, device=x.device)
    else:
        projection = (
            torch.eye(x.shape[1], dtype=x.dtype, device=x.device)
            - covariance_sqrt @ concept_basis @ concept_basis.T @ whitening
        )
    bias = mean_x - projection @ mean_x
    return LeaceEraser(projection, bias, concept_basis.shape[1])
