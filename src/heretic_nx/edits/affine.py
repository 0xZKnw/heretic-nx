"""Affine activation editors, including an exact low-rank LEACE adapter."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from heretic_nx.geometry.leace import LeaceEraser

from .activation_op import ActivationOperator


@dataclass(frozen=True)
class AffineActivationOperator:
    linear: ActivationOperator
    bias: Tensor

    def __post_init__(self) -> None:
        if self.bias.shape != (self.linear.dimension,):
            raise ValueError("affine bias must match the activation dimension")
        if not torch.isfinite(self.bias).all():
            raise ValueError("affine bias must be finite")

    def apply(self, hidden: Tensor, gate: float | Tensor = 1.0) -> Tensor:
        edited = self.linear.apply(hidden, gate=gate)
        gate_tensor = torch.as_tensor(gate, device=hidden.device, dtype=hidden.dtype)
        while gate_tensor.ndim < hidden.ndim:
            gate_tensor = gate_tensor.unsqueeze(-1)
        return edited + self.linear.beta * gate_tensor * self.bias.to(hidden)


def affine_operator_from_leace(
    eraser: LeaceEraser,
    *,
    beta: float = 1.0,
    tolerance: float = 1e-7,
) -> AffineActivationOperator:
    """Factor an exact LEACE affine map into an activation-native operator."""

    projection = eraser.projection.float()
    if projection.ndim != 2 or projection.shape[0] != projection.shape[1]:
        raise ValueError("LEACE projection must be square")
    delta = torch.eye(
        projection.shape[0], device=projection.device, dtype=projection.dtype
    ) - projection
    left, singular, right_t = torch.linalg.svd(delta, full_matrices=False)
    threshold = tolerance * singular.max().clamp_min(torch.finfo(singular.dtype).eps)
    keep = singular > threshold
    roots = singular[keep].sqrt()
    a = left[:, keep] * roots
    b = right_t[keep].T * roots
    return AffineActivationOperator(
        ActivationOperator(a=a, b=b, beta=beta),
        eraser.bias.to(device=projection.device, dtype=projection.dtype),
    )
