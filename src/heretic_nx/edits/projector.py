"""Exact low-rank projector editor and LoRA factorization."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from heretic_nx.geometry.principal_angles import orthonormal_basis


@dataclass(frozen=True)
class ProjectorFactors:
    a: Tensor
    b: Tensor
    beta: float

    @property
    def delta(self) -> Tensor:
        return self.b @ self.a

    def apply_weight(self, weight: Tensor) -> Tensor:
        if weight.shape != self.delta.shape:
            raise ValueError("weight shape does not match projector factors")
        return weight + self.delta.to(weight)


def projector_factors(weight: Tensor, target_basis: Tensor, beta: float) -> ProjectorFactors:
    if not 0.0 <= beta <= 1.0:
        raise ValueError("beta must be in [0, 1]")
    q = orthonormal_basis(target_basis.to(weight))
    if q.shape[0] != weight.shape[0]:
        raise ValueError("target basis must live in the weight output space")
    a = q.T @ weight
    b = -beta * q
    return ProjectorFactors(a=a, b=b, beta=beta)


def input_projector_factors(weight: Tensor, target_basis: Tensor, beta: float) -> ProjectorFactors:
    """Factor ``W(I - beta QQ.T)`` for an input-space projector."""

    if not 0.0 <= beta <= 1.0:
        raise ValueError("beta must be in [0, 1]")
    q = orthonormal_basis(target_basis.to(weight))
    if q.shape[0] != weight.shape[1]:
        raise ValueError("target basis must live in the weight input space")
    a = q.T
    b = -beta * (weight @ q)
    return ProjectorFactors(a=a, b=b, beta=beta)


def apply_projector(outputs: Tensor, target_basis: Tensor, beta: float) -> Tensor:
    q = orthonormal_basis(target_basis.to(outputs))
    return outputs - beta * (outputs @ q) @ q.T
