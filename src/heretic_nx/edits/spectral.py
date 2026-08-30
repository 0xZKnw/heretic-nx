"""Signed spectral editor for target-versus-protected covariance contrasts."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor

from .activation_op import ActivationOperator


@dataclass(frozen=True)
class SignedSpectralEdit:
    operator: ActivationOperator
    basis: Tensor
    coefficients: Tensor
    contrast_eigenvalues: Tensor


def _validated_factor(factor: Tensor, name: str) -> Tensor:
    if factor.ndim != 2:
        raise ValueError(f"{name} must be a rank-2 factor")
    if factor.shape[0] == 0:
        raise ValueError(f"{name} must have a non-empty ambient dimension")
    if not torch.isfinite(factor).all():
        raise ValueError(f"{name} must be finite")
    return factor


def _normalized_factor(factor: Tensor, name: str) -> Tensor:
    if factor.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one direction")
    norm = torch.linalg.vector_norm(factor)
    # The previous dense implementation clamped Gram traces at dtype epsilon.
    # Treat factors below the equivalent norm floor as unsupported instead of
    # amplifying numerical dust into a unit-energy contrast.
    minimum_norm = torch.finfo(factor.dtype).eps**0.5
    if not torch.isfinite(norm) or bool(norm <= minimum_norm):
        raise ValueError(f"{name} is numerically degenerate")
    return factor / norm


def _signed_span_eigh(
    target: Tensor,
    protected: Tensor,
    protected_weight: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Diagonalize the signed contrast inside its exact low-rank span.

    For normalized factors ``T`` and ``P``, the ambient contrast is
    ``T T.T - w P P.T``.  Writing the joined factor as ``Z = Q R`` reduces
    the non-zero eigensystem to the at-most-``rank(T) + rank(P)`` core
    ``R diag(+1, -1) R.T`` without constructing an ambient square matrix.
    """

    target_normalized = _normalized_factor(target, "target_factor")
    pieces = [target_normalized]
    signs = [
        torch.ones(
            target_normalized.shape[1],
            dtype=target.dtype,
            device=target.device,
        )
    ]
    if protected_weight > 0:
        protected_normalized = _normalized_factor(protected, "protected_factor")
        pieces.append(protected_normalized * math.sqrt(protected_weight))
        signs.append(
            -torch.ones(
                protected_normalized.shape[1],
                dtype=target.dtype,
                device=target.device,
            )
        )

    joined = torch.cat(pieces, dim=1)
    signature = torch.cat(signs)
    span_basis, coordinates = torch.linalg.qr(joined, mode="reduced")
    # Reduced QR is not rank revealing.  Collapse numerically null rows of its
    # small coordinate matrix so cancelling or collinear factors cannot turn
    # round-off in an arbitrary QR completion into an eligible direction.
    coordinate_left, coordinate_singular, coordinate_right_t = torch.linalg.svd(
        coordinates,
        full_matrices=False,
    )
    rank_threshold = (
        torch.finfo(coordinate_singular.dtype).eps
        * max(joined.shape)
        * coordinate_singular.max()
    )
    supported = coordinate_singular > rank_threshold
    if not bool(supported.any()):
        raise ValueError("the joined spectral factor span is numerically degenerate")
    span_basis = span_basis @ coordinate_left[:, supported]
    coordinates = (
        coordinate_singular[supported, None] * coordinate_right_t[supported]
    )
    core = (coordinates * signature[None, :]) @ coordinates.T
    core = (core + core.T) / 2
    if not torch.isfinite(core).all():
        raise ValueError("the spectral contrast core became non-finite")
    eigenvalues, core_eigenvectors = torch.linalg.eigh(core)
    numerical_floor = (
        torch.finfo(coordinates.dtype).eps
        * max(coordinates.shape)
        * coordinates.square().sum()
    )
    return eigenvalues, span_basis @ core_eigenvectors, numerical_floor


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
    """Diagonalize a normalized target-minus-protected covariance contrast.

    The eigendecomposition is exact inside the joined factor span and its core
    dimension is bounded by the total factor width, never the ambient model
    width.
    """

    target = _validated_factor(target_factor.float(), "target_factor")
    protected = _validated_factor(
        protected_factor.to(device=target.device, dtype=target.dtype),
        "protected_factor",
    )
    if target.shape[0] != protected.shape[0]:
        raise ValueError("target and protected factors must share an ambient dimension")
    if rank < 1 or not math.isfinite(protected_weight) or protected_weight < 0:
        raise ValueError("rank must be positive and protected_weight non-negative")
    if not math.isfinite(beta) or not 0 <= beta <= 1:
        raise ValueError("beta must be finite and in [0, 1]")
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")
    eigenvalues, eigenvectors, numerical_floor = _signed_span_eigh(
        target,
        protected,
        protected_weight,
    )
    selection_tolerance = max(tolerance, float(numerical_floor.detach()))
    order = torch.argsort(eigenvalues.abs(), descending=True, stable=True)
    if positive_only:
        order = order[eigenvalues[order] > selection_tolerance]
    else:
        order = order[eigenvalues[order].abs() > selection_tolerance]
    selected = order[:rank]
    if selected.numel() == 0:
        raise ValueError("the spectral contrast has no eligible direction")
    values = eigenvalues[selected]
    coefficients = values / values.abs().max().clamp_min(torch.finfo(values.dtype).eps)
    basis = eigenvectors[:, selected]
    # hidden @ B @ A.T = hidden @ Q diag(c) Q.T
    operator = ActivationOperator(a=basis * coefficients, b=basis, beta=beta)
    return SignedSpectralEdit(operator, basis, coefficients, values)
