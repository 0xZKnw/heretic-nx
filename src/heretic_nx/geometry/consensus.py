"""Chordal Grassmann consensus for scenario-specific subspaces."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from .principal_angles import orthonormal_basis


@dataclass(frozen=True)
class ConsensusSubspace:
    basis: Tensor
    eigenvalues: Tensor
    selected_rank: int
    captured_stability_mass: float


def grassmann_consensus(
    bases: Sequence[Tensor],
    *,
    weights: Sequence[float] | None = None,
    eigenvalue_minimum: float = 0.50,
    stability_mass: float = 0.80,
    maximum_rank: int | None = None,
) -> ConsensusSubspace:
    if not bases:
        raise ValueError("at least one subspace is required")
    if not 0 <= eigenvalue_minimum <= 1:
        raise ValueError("eigenvalue_minimum must be in [0, 1]")
    if not 0 < stability_mass <= 1:
        raise ValueError("stability_mass must be in (0, 1]")
    dimension = bases[0].shape[0]
    orthonormal = [orthonormal_basis(basis.float()) for basis in bases]
    if any(basis.shape[0] != dimension for basis in orthonormal):
        raise ValueError("all subspaces must share an ambient dimension")
    if weights is None:
        weight_tensor = torch.ones(len(bases), dtype=orthonormal[0].dtype)
    else:
        if len(weights) != len(bases):
            raise ValueError("weights length must match bases")
        weight_tensor = torch.tensor(weights, dtype=orthonormal[0].dtype)
    if torch.any(weight_tensor < 0) or float(weight_tensor.sum()) <= 0:
        raise ValueError("weights must be non-negative with a positive sum")
    weight_tensor /= weight_tensor.sum()
    factors = [
        basis * weight_tensor[index].sqrt()
        for index, basis in enumerate(orthonormal)
        if basis.shape[1]
    ]
    if not factors:
        empty = bases[0].new_empty((dimension, 0))
        return ConsensusSubspace(empty, bases[0].new_empty((0,)), 0, 0.0)
    joined = torch.cat(factors, dim=1)
    left, singular_values, _ = torch.linalg.svd(joined, full_matrices=False)
    eigenvalues = singular_values.square().clamp(0, 1)
    eligible = int((eigenvalues >= eigenvalue_minimum).sum().item())
    if maximum_rank is not None:
        eligible = min(eligible, maximum_rank)
    if eligible == 0:
        return ConsensusSubspace(left[:, :0], eigenvalues, 0, 0.0)
    total_mass = eigenvalues.sum().clamp_min(torch.finfo(eigenvalues.dtype).eps)
    cumulative = eigenvalues[:eligible].cumsum(dim=0) / total_mass
    mass_rank = int(torch.searchsorted(cumulative, torch.tensor(stability_mass)).item()) + 1
    rank = min(eligible, mass_rank)
    captured = float((eigenvalues[:rank].sum() / total_mass).item())
    return ConsensusSubspace(left[:, :rank], eigenvalues, rank, captured)
