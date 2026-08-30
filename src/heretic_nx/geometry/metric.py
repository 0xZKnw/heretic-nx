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
        if value.device != self.diagonal.device:
            raise ValueError("metric and value must be on the same device")
        if not torch.isfinite(value).all():
            raise ValueError("metric values must be finite")
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
        if covariance_weight < 0 or fisher_weight < 0:
            raise ValueError("metric weights must be non-negative")
        devices = {
            factor.device
            for factor in (covariance_factor, fisher_factor)
            if factor is not None
        }
        if len(devices) > 1:
            raise ValueError("metric factors must be on the same device")
        device = next(iter(devices), torch.device("cpu"))
        pieces = []
        for factor, weight in (
            (covariance_factor, covariance_weight),
            (fisher_factor, fisher_weight),
        ):
            if factor is None or weight == 0:
                continue
            if factor.ndim != 2:
                raise ValueError("metric factors must be rank-2 matrices")
            if factor.shape[1] == 0:
                continue
            value = factor.to(device=device, dtype=dtype)
            if value.shape[0] != dimension:
                raise ValueError("metric factor has the wrong ambient dimension")
            if not torch.isfinite(value).all():
                raise ValueError("metric factors must be finite")
            trace = value.square().sum().clamp_min(torch.finfo(dtype).eps)
            scale = torch.as_tensor(weight * dimension, dtype=dtype, device=device).sqrt()
            pieces.append(value * scale / trace.sqrt())
        joined = (
            torch.cat(pieces, dim=1)
            if pieces
            else torch.empty(dimension, 0, dtype=dtype, device=device)
        )
        return cls(
            torch.full((dimension,), regularization, dtype=dtype, device=device),
            joined,
        )

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
        if not torch.isfinite(activations).all():
            raise ValueError("activations must be finite")
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
    metric_gram = metric.gram(basis)
    gram = (metric_gram + metric_gram.T) / 2
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
        if not torch.isfinite(target).all() or not torch.isfinite(protected).all():
            raise ValueError("geometry inputs must be finite")
        target_basis = metric_orthonormal_basis(target, metric)
        if target_basis.shape[1] == 0:
            return MetricGateResult("reject-site", 90.0, 0.0, target_basis)
        protected_basis = metric_orthonormal_basis(protected, metric)
        if protected_basis.shape[1]:
            gram = metric.gram(protected_basis)
            cross = protected_basis.T @ metric.apply(target_basis)
            raw = target_basis - protected_basis @ torch.linalg.solve(gram, cross)
            cosines = torch.linalg.svdvals(cross).clamp(0, 1)
            minimum_angle = float(
                torch.rad2deg(torch.acos(cosines)).min().item()
            )
        else:
            raw = target_basis
            minimum_angle = 90.0
        retained = float(
            (torch.trace(raw.T @ metric.apply(raw)) / target_basis.shape[1]).item()
        )
        editable = metric_orthonormal_basis(raw, metric)
        if minimum_angle <= self.reject_below_deg or retained < self.reject_energy_below:
            decision = "reject-site"
        elif minimum_angle > self.static_min_deg and retained >= self.static_energy_min:
            decision = "safe-static"
        else:
            decision = "conditional-only"
        return MetricGateResult(decision, minimum_angle, retained, editable)


def require_static_geometry(result: MetricGateResult, *, site_id: str = "unknown") -> Tensor:
    """Return an editable basis only when the static geometry gate passed."""

    if result.decision != "safe-static":
        raise RuntimeError(
            f"site {site_id} is not eligible for a static edit: {result.decision} "
            f"(angle={result.minimum_angle_deg:.6g}, retained={result.retained_energy:.6g})"
        )
    if result.editable_basis.shape[1] == 0 or not torch.isfinite(result.editable_basis).all():
        raise RuntimeError(f"site {site_id} has no finite editable static basis")
    return result.editable_basis
