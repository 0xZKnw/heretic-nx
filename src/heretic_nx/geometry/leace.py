"""Closed-form linear concept erasure baseline.

The fitted eraser is kept as an exact low-rank factorization.  A dense
``projection`` remains available for backwards compatibility, but is only
materialized when callers explicitly request it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import torch
from torch import Tensor


@dataclass(frozen=True, init=False)
class LeaceEraser:
    """An affine LEACE map with an optional low-rank representation.

    ``LeaceEraser(projection, bias, concept_rank)`` remains a supported public
    constructor.  Erasers returned by :func:`fit_leace` instead store
    ``I - projection`` as ``erase_left @ erase_right.T`` so applying the map
    costs ``O(d r)`` rather than ``O(d**2)``.  Accessing ``projection`` lazily
    reconstructs and caches the historical dense tensor.
    """

    bias: Tensor
    concept_rank: int
    _projection: Tensor | None = field(repr=False)
    _erase_left: Tensor | None = field(repr=False)
    _erase_right: Tensor | None = field(repr=False)
    _dimension: int = field(repr=False)

    def __init__(self, projection: Tensor, bias: Tensor, concept_rank: int) -> None:
        """Build an eraser from the historical dense representation."""

        object.__setattr__(self, "bias", bias)
        object.__setattr__(self, "concept_rank", concept_rank)
        object.__setattr__(self, "_projection", projection)
        object.__setattr__(self, "_erase_left", None)
        object.__setattr__(self, "_erase_right", None)
        dimension = projection.shape[0] if projection.ndim else 0
        object.__setattr__(self, "_dimension", dimension)

    @classmethod
    def _from_factors(
        cls,
        erase_left: Tensor,
        erase_right: Tensor,
        bias: Tensor,
        concept_rank: int,
    ) -> "LeaceEraser":
        instance = object.__new__(cls)
        object.__setattr__(instance, "bias", bias)
        object.__setattr__(instance, "concept_rank", concept_rank)
        object.__setattr__(instance, "_projection", None)
        object.__setattr__(instance, "_erase_left", erase_left)
        object.__setattr__(instance, "_erase_right", erase_right)
        object.__setattr__(instance, "_dimension", erase_left.shape[0])
        return instance

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def erase_left(self) -> Tensor | None:
        """Left factor of ``I - projection``, when one is available."""

        return self._erase_left

    @property
    def erase_right(self) -> Tensor | None:
        """Right factor of ``I - projection``, when one is available."""

        return self._erase_right

    @property
    def projection(self) -> Tensor:
        """Return the historical dense projection, materializing it lazily."""

        projection = self._projection
        if projection is None:
            left = self._erase_left
            right = self._erase_right
            if left is None or right is None:  # pragma: no cover - invalid state guard
                raise RuntimeError("LEACE eraser has neither a projection nor factors")
            projection = torch.eye(
                self.dimension, dtype=left.dtype, device=left.device
            ) - left @ right.T
            # Preserve autograd semantics if the factors came from a
            # differentiable fit.  Caching a projection first requested under
            # ``no_grad`` would otherwise permanently detach it.
            if not left.requires_grad and not right.requires_grad:
                object.__setattr__(self, "_projection", projection)
        return projection

    def apply(self, values: Tensor) -> Tensor:
        left = self._erase_left
        right = self._erase_right
        if left is None or right is None:
            return values @ self.projection.T + self.bias
        return values - (values @ right) @ left.T + self.bias


def _concept_coefficients(
    whitened_coefficients: Tensor,
    *,
    ambient_dimension: int,
    tolerance: float,
) -> Tensor:
    """Return concept directions in covariance-support coordinates.

    Multiplication by the right-singular vectors is an isometry.  Running the
    small SVD before that multiplication therefore yields exactly the same
    concept span as the historical ambient-space SVD.  Its rank threshold must
    still use the ambient matrix shape to preserve the old tolerance semantics.
    """

    if whitened_coefficients.shape[0] == 0:
        return whitened_coefficients.new_empty((0, 0))
    if whitened_coefficients.shape[1] == 0:
        return whitened_coefficients.clone()
    left, singular_values, _ = torch.linalg.svd(
        whitened_coefficients, full_matrices=False
    )
    threshold = (
        tolerance
        * max(ambient_dimension, whitened_coefficients.shape[1])
        * singular_values.max()
    )
    keep = singular_values > threshold
    return left[:, keep]


def fit_leace(
    representations: Tensor,
    concepts: Tensor,
    *,
    tolerance: float = 1e-7,
) -> LeaceEraser:
    """Fit LEACE without constructing feature-space covariance matrices.

    For centered ``X = U S V.T``, the historical LEACE erasure term can be
    written exactly as ``L R.T`` where both factors have only the inferred
    concept rank columns.  This thin formulation avoids every ``d x d`` tensor
    during fitting and keeps the final affine map low-rank until a caller asks
    for the backwards-compatible dense :attr:`LeaceEraser.projection`.
    """

    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")
    x = representations.float()
    z = concepts.to(device=x.device, dtype=x.dtype)
    if x.ndim != 2 or z.ndim != 2 or x.shape[0] != z.shape[0]:
        raise ValueError("representations and concepts must be aligned matrices")
    if x.shape[0] < 2:
        raise ValueError("at least two samples are required")
    if x.shape[1] < 1:
        raise ValueError("representations must contain at least one feature")
    if not torch.isfinite(x).all() or not torch.isfinite(z).all():
        raise ValueError("representations and concepts must be finite")
    mean_x = x.mean(dim=0)
    centered_x = x - mean_x
    centered_z = z - z.mean(dim=0)

    left_singular, singular_values, right_t = torch.linalg.svd(
        centered_x, full_matrices=False
    )
    variances = singular_values.square() / x.shape[0]
    threshold = tolerance * variances.max().clamp_min(torch.finfo(x.dtype).eps)
    support = variances > threshold

    support_left = left_singular[:, support]
    support_right = right_t[support].T
    roots = variances[support].sqrt()
    whitened_coefficients = support_left.T @ centered_z / x.shape[0] ** 0.5
    concept_coefficients = _concept_coefficients(
        whitened_coefficients,
        ambient_dimension=x.shape[1],
        tolerance=tolerance,
    )

    if concept_coefficients.shape[1] == 0:
        erase_left = x.new_empty((x.shape[1], 0))
        erase_right = x.new_empty((x.shape[1], 0))
    else:
        erase_left = support_right @ (roots[:, None] * concept_coefficients)
        erase_right = support_right @ (concept_coefficients / roots[:, None])
    bias = erase_left @ (erase_right.T @ mean_x)
    return LeaceEraser._from_factors(
        erase_left,
        erase_right,
        bias,
        concept_coefficients.shape[1],
    )
