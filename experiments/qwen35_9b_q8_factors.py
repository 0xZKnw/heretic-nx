"""Small-sample protected detector fit, without a feature-square covariance."""
import math
import torch


@torch.no_grad()
def fit_detector(target, safe, teacher, protected_weight, ridge_fraction=1e-4):
    if target.ndim != 2 or safe.ndim != 2 or target.shape[1] != safe.shape[1]:
        raise ValueError("incompatible input matrices")
    if min(target.shape) < 1 or safe.shape[0] < 1 or teacher.shape != (target.shape[1],):
        raise ValueError("invalid detector dimensions")
    if not math.isfinite(protected_weight) or protected_weight < 0:
        raise ValueError("invalid protection weight")
    if not math.isfinite(ridge_fraction) or ridge_fraction <= 0:
        raise ValueError("invalid ridge fraction")
    if target.shape[0] + safe.shape[0] > 512:
        raise ValueError("sample-space solver limited to 512 rows")
    t, s, v = (x.detach().cpu().double() for x in (target, safe, teacher))
    if not all(torch.isfinite(x).all() for x in (t, s, v)):
        raise ValueError("non-finite detector inputs")
    diagonal = t.square().mean(0) + protected_weight * s.square().mean(0)
    ridge = ridge_fraction * diagonal.mean().clamp_min(torch.finfo(torch.float32).eps)
    x = torch.cat((t / math.sqrt(len(t)), s * math.sqrt(protected_weight / len(s))))
    y = torch.cat((t @ v / math.sqrt(len(t)), torch.zeros(len(s), dtype=torch.float64)))
    gram = x @ x.T
    gram.diagonal().add_(ridge)
    solution = (x.T @ torch.linalg.solve(gram, y)).float()
    exported = solution.double()
    rhs = t.T @ (t @ v) / len(t)
    residual = rhs - (t.T @ (t @ exported) / len(t)
                     + protected_weight * s.T @ (s @ exported) / len(s) + ridge * exported)
    ratio = float(residual.norm() / rhs.norm().clamp_min(torch.finfo(torch.float64).tiny))
    if not math.isfinite(ratio) or ratio > 1e-5:
        raise RuntimeError(f"exported detector residual too high: {ratio}")
    return solution, {"residual_ratio": ratio, "ridge": float(ridge),
                      "solver": "sample-space-fp64-export-checked", "converged": True}
