"""Small CPU QCQP solver for interaction-aware PRIME intensity allocation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.optimize import minimize
from torch import Tensor


@dataclass(frozen=True)
class QCQPResult:
    beta: Tensor
    objective: float
    capability_cost: float
    risk_cost: float
    success: bool
    message: str
    iterations: int


def _symmetric(value: Tensor) -> Tensor:
    return (value.float() + value.float().T) / 2


def solve_qcqp(
    gain: Tensor,
    benign_hessian: Tensor,
    capability_metric: Tensor,
    risk_metric: Tensor,
    *,
    capability_budget: float,
    risk_budget: float,
    beta_max: float | Tensor = 0.8,
    max_iterations: int = 500,
    tolerance: float = 1e-8,
) -> QCQPResult:
    """Solve a convex reduced QCQP with explicit H/C ellipsoidal constraints."""

    g = gain.detach().float().cpu()
    h = _symmetric(benign_hessian).cpu()
    fc = _symmetric(capability_metric).cpu()
    fh = _symmetric(risk_metric).cpu()
    dimension = g.numel()
    if g.ndim != 1 or any(matrix.shape != (dimension, dimension) for matrix in (h, fc, fh)):
        raise ValueError("QCQP tensors have incompatible shapes")
    if capability_budget < 0 or risk_budget < 0:
        raise ValueError("budgets must be non-negative")
    maximum = torch.as_tensor(beta_max, dtype=torch.float32).expand(dimension).cpu()
    if torch.any(maximum <= 0):
        raise ValueError("beta_max must be positive")

    h_np, fc_np, fh_np, g_np = (item.numpy().astype(np.float64) for item in (h, fc, fh, g))

    def objective(beta: np.ndarray) -> float:
        return float(0.5 * beta @ h_np @ beta - g_np @ beta)

    def objective_jac(beta: np.ndarray) -> np.ndarray:
        return h_np @ beta - g_np

    constraints = [
        {
            "type": "ineq",
            "fun": lambda beta: capability_budget - beta @ fc_np @ beta,
            "jac": lambda beta: -2 * fc_np @ beta,
        },
        {
            "type": "ineq",
            "fun": lambda beta: risk_budget - beta @ fh_np @ beta,
            "jac": lambda beta: -2 * fh_np @ beta,
        },
    ]
    result = minimize(
        objective,
        np.zeros(dimension, dtype=np.float64),
        jac=objective_jac,
        bounds=[(-float(limit), float(limit)) for limit in maximum],
        constraints=constraints,
        method="SLSQP",
        options={"maxiter": max_iterations, "ftol": tolerance, "disp": False},
    )
    beta = torch.from_numpy(result.x).float()
    capability_cost = float(beta @ fc @ beta)
    risk_cost = float(beta @ fh @ beta)
    # Numerical fail-closed projection. Uniform scaling preserves both constraints.
    scale = 1.0
    if capability_cost > capability_budget + tolerance and capability_cost > 0:
        scale = min(scale, (capability_budget / capability_cost) ** 0.5)
    if risk_cost > risk_budget + tolerance and risk_cost > 0:
        scale = min(scale, (risk_budget / risk_cost) ** 0.5)
    if scale < 1:
        beta *= scale
        capability_cost = float(beta @ fc @ beta)
        risk_cost = float(beta @ fh @ beta)
    feasible = (
        capability_cost <= capability_budget + 10 * tolerance
        and risk_cost <= risk_budget + 10 * tolerance
        and bool(torch.all(beta.abs() <= maximum + 10 * tolerance))
    )
    return QCQPResult(
        beta=beta,
        objective=float(0.5 * beta @ h @ beta - g @ beta),
        capability_cost=capability_cost,
        risk_cost=risk_cost,
        success=bool(result.success and feasible),
        message=str(result.message),
        iterations=int(result.nit),
    )
