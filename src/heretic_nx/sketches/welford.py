"""Mergeable streaming first and diagonal second moments."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class WelfordState:
    count: int
    mean: Tensor
    m2: Tensor

    @classmethod
    def empty(
        cls,
        dimension: int,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> "WelfordState":
        zeros = torch.zeros(dimension, device=device, dtype=dtype)
        return cls(count=0, mean=zeros.clone(), m2=zeros.clone())

    def update(self, values: Tensor) -> None:
        batch = values.detach().reshape(-1, self.mean.numel()).to(
            device=self.mean.device,
            dtype=self.mean.dtype,
        )
        if batch.numel() == 0:
            return
        batch_count = batch.shape[0]
        batch_mean = batch.mean(dim=0)
        centered = batch - batch_mean
        batch_m2 = (centered * centered).sum(dim=0)
        self._merge_raw(batch_count, batch_mean, batch_m2)

    def merge(self, other: "WelfordState") -> None:
        if self.mean.shape != other.mean.shape:
            raise ValueError("Cannot merge Welford states with different dimensions")
        self._merge_raw(
            other.count,
            other.mean.to(self.mean),
            other.m2.to(self.m2),
        )

    def _merge_raw(self, count: int, mean: Tensor, m2: Tensor) -> None:
        if count == 0:
            return
        if self.count == 0:
            self.count = count
            self.mean.copy_(mean)
            self.m2.copy_(m2)
            return
        total = self.count + count
        delta = mean - self.mean
        self.mean.add_(delta * (count / total))
        self.m2.add_(m2 + delta.square() * (self.count * count / total))
        self.count = total

    @property
    def variance(self) -> Tensor:
        if self.count < 2:
            return torch.zeros_like(self.m2)
        return self.m2 / (self.count - 1)
