"""Protected-subspace projection and principal-angle admission gates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor


def orthonormal_basis(matrix: Tensor, tolerance: float = 1e-7) -> Tensor:
    if matrix.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    if matrix.shape[1] == 0:
        return matrix.clone()
    u, singular_values, _ = torch.linalg.svd(matrix, full_matrices=False)
    threshold = tolerance * max(matrix.shape) * singular_values.max()
    keep = singular_values > threshold
    return u[:, keep]


def project_out(target: Tensor, protected: Tensor) -> Tensor:
    p = orthonormal_basis(protected)
    if p.shape[1] == 0:
        return target.clone()
    return target - p @ (p.T @ target)


def principal_angles_deg(left: Tensor, right: Tensor) -> Tensor:
    q_left = orthonormal_basis(left)
    q_right = orthonormal_basis(right)
    if q_left.shape[1] == 0 or q_right.shape[1] == 0:
        return torch.tensor([90.0], device=left.device, dtype=left.dtype)
    cosines = torch.linalg.svdvals(q_left.T @ q_right).clamp(0, 1)
    return torch.rad2deg(torch.acos(cosines))


@dataclass(frozen=True)
class GeometryGateResult:
    decision: Literal["safe-static", "conditional-only", "reject-site"]
    minimum_angle_deg: float
    retained_energy: float
    editable_basis: Tensor


@dataclass(frozen=True)
class GeometryGate:
    static_min_deg: float = 45.0
    reject_below_deg: float = 20.0
    static_energy_min: float = 0.60
    reject_energy_below: float = 0.20

    def evaluate(self, target: Tensor, protected: Tensor) -> GeometryGateResult:
        target_basis = orthonormal_basis(target)
        projected = project_out(target_basis, protected)
        editable = orthonormal_basis(projected)
        target_energy = target_basis.square().sum().clamp_min(torch.finfo(target.dtype).eps)
        retained = float((projected.square().sum() / target_energy).item())
        minimum_angle = float(principal_angles_deg(protected, target_basis).min().item())

        if minimum_angle <= self.reject_below_deg or retained < self.reject_energy_below:
            decision = "reject-site"
        elif minimum_angle > self.static_min_deg and retained >= self.static_energy_min:
            decision = "safe-static"
        else:
            decision = "conditional-only"
        return GeometryGateResult(decision, minimum_angle, retained, editable)
