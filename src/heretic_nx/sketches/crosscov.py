"""Mergeable streaming cross-covariance statistics."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class CrossCovarianceState:
    count: int
    mean_x: Tensor
    mean_z: Tensor
    co_moment: Tensor

    @classmethod
    def empty(
        cls,
        x_dimension: int,
        z_dimension: int,
        *,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str = "cpu",
    ) -> "CrossCovarianceState":
        return cls(
            0,
            torch.zeros(x_dimension, dtype=dtype, device=device),
            torch.zeros(z_dimension, dtype=dtype, device=device),
            torch.zeros(x_dimension, z_dimension, dtype=dtype, device=device),
        )

    def update(self, x: Tensor, z: Tensor) -> None:
        x = x.to(device=self.mean_x.device, dtype=self.mean_x.dtype)
        z = z.to(device=self.mean_z.device, dtype=self.mean_z.dtype)
        if x.ndim == 1:
            x = x[None, :]
        if z.ndim == 1:
            z = z[None, :]
        if x.ndim != 2 or z.ndim != 2 or x.shape[0] != z.shape[0]:
            raise ValueError("x and z must be matrices with equal sample counts")
        if x.shape[1] != self.mean_x.numel() or z.shape[1] != self.mean_z.numel():
            raise ValueError("feature dimensions do not match the state")
        if x.shape[0] == 0:
            return
        other = CrossCovarianceState.empty(
            x.shape[1], z.shape[1], dtype=x.dtype, device=x.device
        )
        other.count = x.shape[0]
        other.mean_x = x.mean(dim=0)
        other.mean_z = z.mean(dim=0)
        other.co_moment = (x - other.mean_x).T @ (z - other.mean_z)
        self.merge(other)

    def merge(self, other: "CrossCovarianceState") -> None:
        if other.count == 0:
            return
        if self.count == 0:
            self.count = other.count
            self.mean_x = other.mean_x.clone()
            self.mean_z = other.mean_z.clone()
            self.co_moment = other.co_moment.clone()
            return
        if self.mean_x.shape != other.mean_x.shape or self.mean_z.shape != other.mean_z.shape:
            raise ValueError("states have incompatible dimensions")
        total = self.count + other.count
        delta_x = other.mean_x - self.mean_x
        delta_z = other.mean_z - self.mean_z
        correction = torch.outer(delta_x, delta_z) * (self.count * other.count / total)
        self.co_moment = self.co_moment + other.co_moment + correction
        self.mean_x = self.mean_x + delta_x * (other.count / total)
        self.mean_z = self.mean_z + delta_z * (other.count / total)
        self.count = total

    @property
    def covariance(self) -> Tensor:
        if self.count < 2:
            return torch.zeros_like(self.co_moment)
        return self.co_moment / (self.count - 1)
