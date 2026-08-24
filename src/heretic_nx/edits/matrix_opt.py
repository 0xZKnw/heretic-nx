"""Clean-room low-rank matrix editor optimized on target/protected activations."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
import torch.nn.functional as F

from .activation_op import ActivationOperator


@dataclass(frozen=True)
class LowRankOptimizationResult:
    operator: ActivationOperator
    initial_loss: float
    final_loss: float
    protected_relative_drift: float
    target_separation_ratio: float
    steps: int


def fit_low_rank_matrix_operator(
    target_activations: Tensor,
    protected_activations: Tensor,
    *,
    rank: int,
    beta: float,
    steps: int = 200,
    learning_rate: float = 0.03,
    protected_weight: float = 4.0,
    operator_weight: float = 0.01,
    seed: int = 0,
) -> LowRankOptimizationResult:
    """Fit ``I - beta A B.T`` while penalizing protected activation drift."""

    target = target_activations.float()
    protected = protected_activations.to(device=target.device, dtype=target.dtype)
    if target.ndim != 2 or protected.ndim != 2 or target.shape[1] != protected.shape[1]:
        raise ValueError("target and protected activations must be compatible matrices")
    if min(target.shape[0], protected.shape[0]) < 2:
        raise ValueError("at least two target and protected rows are required")
    if not torch.isfinite(target).all() or not torch.isfinite(protected).all():
        raise ValueError("optimization activations must be finite")
    if rank < 1 or rank > target.shape[1] or steps < 1 or learning_rate <= 0:
        raise ValueError("rank, steps, or learning_rate is invalid")
    if protected_weight < 0 or operator_weight < 0:
        raise ValueError("regularization weights must be non-negative")

    generator = torch.Generator(device=target.device).manual_seed(seed)
    separation = target.mean(dim=0) - protected.mean(dim=0)
    first = F.normalize(separation, dim=0)
    initial = torch.randn(
        target.shape[1], rank, generator=generator, device=target.device, dtype=target.dtype
    )
    initial[:, 0] = first
    q, _ = torch.linalg.qr(initial, mode="reduced")
    a = torch.nn.Parameter(q.clone())
    b = torch.nn.Parameter(q.clone())
    optimizer = torch.optim.AdamW((a, b), lr=learning_rate, weight_decay=0.0)
    baseline_separation = separation.square().mean().clamp_min(torch.finfo(target.dtype).eps)
    protected_scale = protected.square().mean().clamp_min(torch.finfo(target.dtype).eps)

    def objective() -> tuple[Tensor, Tensor, Tensor]:
        target_delta = (target @ b) @ a.T
        protected_delta = (protected @ b) @ a.T
        edited_separation = (target - beta * target_delta).mean(dim=0) - protected.mean(dim=0)
        separation_loss = edited_separation.square().mean() / baseline_separation
        protected_loss = beta**2 * protected_delta.square().mean() / protected_scale
        matrix = a @ b.T
        loss = separation_loss + protected_weight * protected_loss + operator_weight * matrix.square().mean()
        return loss, protected_loss, separation_loss

    initial_loss = float(objective()[0].detach().item())
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss, _protected_loss, _separation_loss = objective()
        if not torch.isfinite(loss):
            raise RuntimeError("low-rank optimization became non-finite")
        loss.backward()
        torch.nn.utils.clip_grad_norm_((a, b), 1.0)
        optimizer.step()
        with torch.no_grad():
            # Bound the spectral scale so beta retains its [0, 1] meaning.
            norm = torch.linalg.matrix_norm(a @ b.T, ord=2).clamp_min(1.0)
            a.div_(norm.sqrt())
            b.div_(norm.sqrt())

    final_loss, protected_loss, separation_loss = objective()
    return LowRankOptimizationResult(
        operator=ActivationOperator(a.detach(), b.detach(), beta),
        initial_loss=initial_loss,
        final_loss=float(final_loss.detach().item()),
        protected_relative_drift=float(protected_loss.detach().sqrt().item()),
        target_separation_ratio=float(separation_loss.detach().sqrt().item()),
        steps=steps,
    )
