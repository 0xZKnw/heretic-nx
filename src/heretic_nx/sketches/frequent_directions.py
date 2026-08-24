"""Deterministic Frequent Directions covariance sketch."""

from __future__ import annotations

import torch
from torch import Tensor


class FrequentDirections:
    def __init__(
        self,
        rank: int,
        dimension: int,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if rank < 1 or rank > dimension:
            raise ValueError("rank must be in [1, dimension]")
        self.rank = rank
        self.dimension = dimension
        self.device = torch.device(device)
        self.dtype = dtype
        self._rows = torch.empty((0, dimension), device=self.device, dtype=dtype)
        self.samples_seen = 0

    @property
    def rows(self) -> Tensor:
        return self._compress(self._rows, shrink=False)

    def update(self, values: Tensor) -> None:
        batch = values.detach().reshape(-1, self.dimension).to(
            device=self.device,
            dtype=self.dtype,
        )
        self.samples_seen += batch.shape[0]
        self._rows = torch.cat((self._rows, batch), dim=0)
        if self._rows.shape[0] >= 2 * self.rank:
            self._rows = self._compress(self._rows, shrink=True)

    def merge(self, other: "FrequentDirections") -> None:
        if (self.rank, self.dimension) != (other.rank, other.dimension):
            raise ValueError("Cannot merge sketches with different shapes")
        self.update(other.rows)
        self.samples_seen += other.samples_seen - other.rows.shape[0]

    def covariance_approx(self) -> Tensor:
        rows = self.rows
        return rows.T @ rows

    def basis(self, rank: int | None = None) -> Tensor:
        requested = min(rank or self.rank, self.rank, self.dimension)
        rows = self.rows
        if rows.numel() == 0:
            return torch.empty(
                (self.dimension, 0), device=self.device, dtype=self.dtype
            )
        _, singular_values, vh = torch.linalg.svd(rows, full_matrices=False)
        keep = min(requested, int((singular_values > 0).sum().item()))
        return vh[:keep].T.contiguous()

    def _compress(self, rows: Tensor, *, shrink: bool) -> Tensor:
        if rows.shape[0] <= self.rank:
            return rows
        _, singular_values, vh = torch.linalg.svd(rows, full_matrices=False)
        keep = min(self.rank, singular_values.numel())
        retained = singular_values[:keep].square()
        if shrink and singular_values.numel() >= self.rank:
            delta = singular_values[self.rank - 1].square()
            retained = torch.clamp(retained - delta, min=0)
        return torch.diag(torch.sqrt(retained)) @ vh[:keep]
