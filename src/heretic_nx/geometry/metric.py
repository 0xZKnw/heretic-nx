"""Low-rank covariance/Fisher metric and M-orthogonal geometry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

from .principal_angles import orthonormal_basis


@dataclass(frozen=True)
class LowRankMetric:
    diagonal: Tensor
    factors: Tensor

    @property
    def dimension(self) -> int:
        return self.diagonal.numel()

    def apply(self, value: Tensor) -> Tensor:
        if value.shape[0] != self.dimension:
            raise ValueError("metric and value dimensions differ")
        result = self.diagonal * value if value.ndim == 1 else self.diagonal[:, None] * value
        if self.factors.shape[1]:
            result = result + self.factors @ (self.factors.T @ value)
        return result

    def gram(self, left: Tensor, right: Tensor | None = None) -> Tensor:
        right = left if right is None else right
        return left.T @ self.apply(right)

    def dense(self) -> Tensor:
        return torch.diag(self.diagonal) + self.factors @ self.factors.T

    @classmethod
    def from_factors(
        cls,
        dimension: int,
        *,
        covariance_factor: Tensor | None = None,
        fisher_factor: Tensor | None = None,
        regularization: float = 1e-3,
        covariance_weight: float = 1.0,
        fisher_weight: float = 1.0,
        dtype: torch.dtype = torch.float32,
    ) -> "LowRankMetric":
        if regularization <= 0:
            raise ValueError("regularization must be positive")
        pieces = []
        for factor, weight in (
            (covariance_factor, covariance_weight),
            (fisher_factor, fisher_weight),
        ):
            if factor is None or factor.shape[1] == 0 or weight == 0:
                continue
            value = factor.to(dtype=dtype)
            if value.shape[0] != dimension:
                raise ValueError("metric factor has the wrong ambient dimension")
            trace = value.square().sum().clamp_min(torch.finfo(dtype).eps)
            pieces.append(value * torch.sqrt(torch.tensor(weight * dimension, dtype=dtype) / trace))
        joined = torch.cat(pieces, dim=1) if pieces else torch.empty(dimension, 0, dtype=dtype)
        return cls(torch.full((dimension,), regularization, dtype=dtype), joined)

    @classmethod
    def from_samples(
        cls,
        activations: Tensor,
        *,
        fisher_factor: Tensor | None = None,
        regularization: float = 1e-3,
        covariance_weight: float = 1.0,
        fisher_weight: float = 1.0,
    ) -> "LowRankMetric":
        if activations.ndim != 2 or activations.shape[0] < 2:
            raise ValueError("at least two activation rows are required")
        centered = activations.float() - activations.float().mean(dim=0)
        covariance_factor = centered.T / (activations.shape[0] - 1) ** 0.5
        return cls.from_factors(
            activations.shape[1],
            covariance_factor=covariance_factor,
            fisher_factor=fisher_factor,
            regularization=regularization,
            covariance_weight=covariance_weight,
            fisher_weight=fisher_weight,
        )


def metric_orthonormal_basis(
    matrix: Tensor,
    metric: LowRankMetric,
    tolerance: float = 1e-7,
) -> Tensor:
    basis = orthonormal_basis(matrix.float(), tolerance=tolerance)
    if basis.shape[1] == 0:
        return basis
    gram = (metric.gram(basis) + metric.gram(basis).T) / 2
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    threshold = tolerance * eigenvalues.abs().max().clamp_min(torch.finfo(gram.dtype).eps)
    keep = eigenvalues > threshold
    return basis @ (eigenvectors[:, keep] / eigenvalues[keep].sqrt())


def metric_residualize(
    target: Tensor,
    protected: Tensor,
    metric: LowRankMetric,
    *,
    ridge: float = 1e-6,
) -> Tensor:
    target_basis = metric_orthonormal_basis(target, metric)
    protected_basis = metric_orthonormal_basis(protected, metric)
    if protected_basis.shape[1] == 0:
        return target_basis
    gram = metric.gram(protected_basis)
    cross = protected_basis.T @ metric.apply(target_basis)
    coefficients = torch.linalg.solve(
        gram + ridge * torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device),
        cross,
    )
    residual = target_basis - protected_basis @ coefficients
    return metric_orthonormal_basis(residual, metric)


def metric_principal_angles_deg(left: Tensor, right: Tensor, metric: LowRankMetric) -> Tensor:
    q_left = metric_orthonormal_basis(left, metric)
    q_right = metric_orthonormal_basis(right, metric)
    if q_left.shape[1] == 0 or q_right.shape[1] == 0:
        return torch.tensor([90.0], dtype=metric.diagonal.dtype)
    cosines = torch.linalg.svdvals(q_left.T @ metric.apply(q_right)).clamp(0, 1)
    return torch.rad2deg(torch.acos(cosines))


@dataclass(frozen=True)
class MetricGateResult:
    decision: Literal["safe-static", "conditional-only", "reject-site"]
    minimum_angle_deg: float
    retained_energy: float
    editable_basis: Tensor


@dataclass(frozen=True)
class MetricGeometryGate:
    static_min_deg: float = 45.0
    reject_below_deg: float = 20.0
    static_energy_min: float = 0.60
    reject_energy_below: float = 0.20

    def evaluate(
        self,
        target: Tensor,
        protected: Tensor,
        metric: LowRankMetric,
    ) -> MetricGateResult:
        target_basis = metric_orthonormal_basis(target, metric)
        if target_basis.shape[1] == 0:
            return MetricGateResult("reject-site", 90.0, 0.0, target_basis)
        protected_basis = metric_orthonormal_basis(protected, metric)
        if protected_basis.shape[1]:
            gram = metric.gram(protected_basis)
            cross = protected_basis.T @ metric.apply(target_basis)
            raw = target_basis - protected_basis @ torch.linalg.solve(gram, cross)
        else:
            raw = target_basis
        retained = float(
            (torch.trace(raw.T @ metric.apply(raw)) / target_basis.shape[1]).item()
        )
        editable = metric_orthonormal_basis(raw, metric)
        minimum_angle = float(metric_principal_angles_deg(protected, target, metric).min().item())
        if minimum_angle <= self.reject_below_deg or retained < self.reject_energy_below:
            decision = "reject-site"
        elif minimum_angle > self.static_min_deg and retained >= self.static_energy_min:
            decision = "safe-static"
        else:
            decision = "conditional-only"
        return MetricGateResult(decision, minimum_angle, retained, editable)
