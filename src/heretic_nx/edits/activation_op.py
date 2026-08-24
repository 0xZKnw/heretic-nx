"""Canonical activation-native low-rank operator."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
from torch import Tensor

from heretic_nx.geometry.metric import LowRankMetric


@dataclass(frozen=True)
class ActivationOperator:
    """Column-space operator ``h' = h - gate * beta * A(B.T h)``."""

    a: Tensor
    b: Tensor
    beta: float
    sparse_index: Tensor | None = None

    def __post_init__(self) -> None:
        if self.a.ndim != 2 or self.b.ndim != 2 or self.a.shape != self.b.shape:
            raise ValueError("a and b must have the same (dimension, rank) shape")
        if not 0 <= self.beta <= 1:
            raise ValueError("beta must be in [0, 1]")
        if self.sparse_index is not None:
            if self.sparse_index.ndim != 1:
                raise ValueError("sparse_index must be one-dimensional")
            if self.sparse_index.numel() and (
                int(self.sparse_index.min()) < 0
                or int(self.sparse_index.max()) >= self.a.shape[0]
            ):
                raise ValueError("sparse_index contains an invalid ambient coordinate")

    @property
    def dimension(self) -> int:
        return self.a.shape[0]

    @property
    def rank(self) -> int:
        return self.a.shape[1]

    def apply(self, hidden: Tensor, gate: float | Tensor = 1.0) -> Tensor:
        if hidden.shape[-1] != self.dimension:
            raise ValueError("hidden state has the wrong final dimension")
        a = self.a.to(hidden)
        b = self.b.to(hidden)
        delta = (hidden @ b) @ a.T
        if self.sparse_index is not None:
            mask = torch.zeros(self.dimension, dtype=hidden.dtype, device=hidden.device)
            mask[self.sparse_index.to(hidden.device)] = 1
            delta = delta * mask
        gate_tensor = torch.as_tensor(gate, device=hidden.device, dtype=hidden.dtype)
        while gate_tensor.ndim < delta.ndim:
            gate_tensor = gate_tensor.unsqueeze(-1)
        return hidden - self.beta * gate_tensor * delta

    def with_sparse_index(self, sparse_index: Tensor | None) -> "ActivationOperator":
        return replace(self, sparse_index=sparse_index)


def metric_projector_operator(
    basis: Tensor,
    metric: LowRankMetric,
    beta: float,
    *,
    ridge: float = 1e-6,
) -> ActivationOperator:
    q = basis.float()
    if q.ndim != 2 or q.shape[0] != metric.dimension:
        raise ValueError("basis must match the metric dimension")
    gram = q.T @ metric.apply(q)
    inverse_gram = torch.linalg.inv(
        gram + ridge * torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
    )
    # B.T = G^-1 Q.T M, hence B = M Q G^-1 for symmetric M and G.
    b = metric.apply(q) @ inverse_gram
    return ActivationOperator(a=q, b=b, beta=beta)


def activation_forward_hook(operator: ActivationOperator, gate: float | Tensor = 1.0):
    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if not torch.is_tensor(hidden):
            return output
        edited = operator.apply(hidden, gate=gate)
        if isinstance(output, tuple):
            return (edited,) + output[1:]
        return edited

    return hook
