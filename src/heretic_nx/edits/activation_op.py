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
            if self.sparse_index.dtype not in {
                torch.uint8,
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
            }:
                raise ValueError("sparse_index must contain integer coordinates")
            if self.sparse_index.numel() and (
                int(self.sparse_index.min()) < 0
                or int(self.sparse_index.max()) >= self.a.shape[0]
            ):
                raise ValueError("sparse_index contains an invalid ambient coordinate")
            # The previous mask semantics ignored duplicate coordinates. Canonicalize
            # them once so the indexed kernel preserves that behavior without adding
            # duplicate updates on every forward pass.
            object.__setattr__(
                self,
                "sparse_index",
                torch.unique(self.sparse_index.to(dtype=torch.long), sorted=True),
            )

    @property
    def dimension(self) -> int:
        return self.a.shape[0]

    @property
    def rank(self) -> int:
        return self.a.shape[1]

    def apply(self, hidden: Tensor, gate: float | Tensor = 1.0) -> Tensor:
        if hidden.shape[-1] != self.dimension:
            raise ValueError("hidden state has the wrong final dimension")
        b = self.b.to(hidden)
        gate_tensor = torch.as_tensor(gate, device=hidden.device, dtype=hidden.dtype)
        while gate_tensor.ndim < hidden.ndim:
            gate_tensor = gate_tensor.unsqueeze(-1)

        if self.sparse_index is not None:
            sparse_size = self.sparse_index.numel()
            if sparse_size == 0:
                return hidden.clone()
            if sparse_size * 2 <= self.dimension:
                # Sparse indices select output coordinates, not latent inputs. Avoid
                # materializing the ambient-width delta and scatter the compact update
                # into a fresh output tensor instead.
                source_index = self.sparse_index.to(device=self.a.device)
                output_index = self.sparse_index.to(device=hidden.device)
                selected_a = self.a.index_select(0, source_index).to(hidden)
                delta = (hidden @ b) @ selected_a.T
                return hidden.index_add(
                    -1,
                    output_index,
                    -self.beta * gate_tensor * delta,
                )

        a = self.a.to(hidden)
        delta = (hidden @ b) @ a.T
        if self.sparse_index is not None and sparse_size < self.dimension:
            # Dense selections are faster as one ambient GEMM plus a mask than as a
            # large indexed scatter. This branch retains the previous kernel around
            # the measured half-dimension crossover.
            mask = torch.zeros(self.dimension, dtype=hidden.dtype, device=hidden.device)
            mask[self.sparse_index.to(hidden.device)] = 1
            delta = delta * mask
        return hidden - self.beta * gate_tensor * delta

    def with_sparse_index(self, sparse_index: Tensor | None) -> "ActivationOperator":
        return replace(self, sparse_index=sparse_index)

    def spectral_norm(self) -> float:
        """Return ``||A B.T||_2`` from a rank-sized core matrix."""

        if self.rank == 0:
            return 0.0
        _qa, ra = torch.linalg.qr(self.a.float(), mode="reduced")
        _qb, rb = torch.linalg.qr(self.b.float(), mode="reduced")
        return float(torch.linalg.svdvals(ra @ rb.T).max().item())

    def bounded(self, maximum_norm: float = 1.0) -> "ActivationOperator":
        """Scale the low-rank map to an auditable Euclidean spectral bound."""

        if maximum_norm <= 0:
            raise ValueError("maximum_norm must be positive")
        norm = self.spectral_norm()
        if not torch.isfinite(torch.tensor(norm)):
            raise ValueError("operator spectral norm must be finite")
        if norm <= maximum_norm:
            return self
        return replace(self, a=self.a * (maximum_norm / norm))


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
    metric_q = metric.apply(q)
    gram = q.T @ metric_q
    inverse_gram = torch.linalg.inv(
        gram + ridge * torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
    )
    # B.T = G^-1 Q.T M, hence B = M Q G^-1 for symmetric M and G.
    b = metric_q @ inverse_gram
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
