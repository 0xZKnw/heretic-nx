"""Exact low-rank Cayley rotation without materializing a d x d inverse."""

from __future__ import annotations

import torch
from torch import Tensor


def _validate(u: Tensor, v: Tensor) -> None:
    if u.ndim != 2 or v.ndim != 2 or u.shape != v.shape:
        raise ValueError("u and v must have the same [dimension, rank] shape")


def skew_factors(u: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
    _validate(u, v)
    rank = u.shape[1]
    identity = torch.eye(rank, device=u.device, dtype=u.dtype)
    zeros = torch.zeros_like(identity)
    z = torch.cat((u, v), dim=1)
    j = torch.cat(
        (torch.cat((zeros, identity), dim=1), torch.cat((-identity, zeros), dim=1)),
        dim=0,
    )
    return z, j


def apply_cayley(values: Tensor, u: Tensor, v: Tensor) -> Tensor:
    """Apply R=(I-A)(I+A)^-1 to row vectors in ``values``."""
    _validate(u, v)
    if values.shape[-1] != u.shape[0]:
        raise ValueError("values and Cayley factors use different dimensions")
    z, j = skew_factors(u.to(values), v.to(values))
    flat = values.reshape(-1, values.shape[-1]).T
    middle = -j + z.T @ z  # J^-1 = -J
    solved = flat - z @ torch.linalg.solve(middle, z.T @ flat)
    skew_times_solved = u.to(values) @ (v.to(values).T @ solved) - v.to(values) @ (
        u.to(values).T @ solved
    )
    rotated = solved - skew_times_solved
    return rotated.T.reshape_as(values)


def cayley_matrix(u: Tensor, v: Tensor) -> Tensor:
    _validate(u, v)
    dimension = u.shape[0]
    identity = torch.eye(dimension, device=u.device, dtype=u.dtype)
    return apply_cayley(identity, u, v).T
