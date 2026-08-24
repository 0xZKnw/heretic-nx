"""Signed spectral editor for target-versus-protected covariance contrasts."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .activation_op import ActivationOperator


@dataclass(frozen=True)
class SignedSpectralEdit:
    operator: ActivationOperator
    basis: Tensor
    coefficients: Tensor
    contrast_eigenvalues: Tensor


def _normalized_gram(factor: Tensor, name: str) -> Tensor:
    if factor.ndim != 2:
        raise ValueError(f"{name} must be a rank-2 factor")
    if not torch.isfinite(factor).all():
        raise ValueError(f"{name} must be finite")
    gram = factor @ factor.T
    return gram / torch.trace(gram).clamp_min(torch.finfo(gram.dtype).eps)


def fit_signed_spectral_operator(
    target_factor: Tensor,
    protected_factor: Tensor,
    *,
    rank: int,
    beta: float,
    protected_weight: float = 1.0,
    positive_only: bool = True,
    tolerance: float = 1e-7,
) -> SignedSpectralEdit:
    """Diagonalize a normalized target-minus-protected covariance contrast."""

    target = target_factor.float()
    protected = protected_factor.to(device=target.device, dtype=target.dtype)
    if target.shape[0] != protected.shape[0]:
        raise ValueError("target and protected factors must share an ambient dimension")
    if rank < 1 or protected_weight < 0:
        raise ValueError("rank must be positive and protected_weight non-negative")
    contrast = _normalized_gram(target, "target_factor") - protected_weight * _normalized_gram(
        protected, "protected_factor"
    )
    contrast = (contrast + contrast.T) / 2
    eigenvalues, eigenvectors = torch.linalg.eigh(contrast)
    order = torch.argsort(eigenvalues.abs(), descending=True, stable=True)
    if positive_only:
        order = order[eigenvalues[order] > tolerance]
    else:
        order = order[eigenvalues[order].abs() > tolerance]
    selected = order[:rank]
    if selected.numel() == 0:
        raise ValueError("the spectral contrast has no eligible direction")
    values = eigenvalues[selected]
    coefficients = values / values.abs().max().clamp_min(torch.finfo(values.dtype).eps)
    basis = eigenvectors[:, selected]
    # hidden @ B @ A.T = hidden @ Q diag(c) Q.T
    operator = ActivationOperator(a=basis * coefficients, b=basis, beta=beta)
    return SignedSpectralEdit(operator, basis, coefficients, values)
