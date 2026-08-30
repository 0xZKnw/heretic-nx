"""Deterministic exact PCA in the smaller sample or feature Gram space."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor


@dataclass(frozen=True)
class PrincipalComponentFit:
    basis: Tensor
    singular_values: Tensor
    effective_rank: int
    retained_energy_fraction: float


def exact_principal_components(
    samples: Tensor,
    *,
    maximum_rank: int,
    center: bool = True,
    tolerance: float = 1e-7,
) -> PrincipalComponentFit:
    """Return exact leading right-singular vectors without randomized null axes."""

    if samples.ndim != 2 or samples.shape[0] < 2 or samples.shape[1] < 1:
        raise ValueError("PCA samples must be a matrix with at least two rows")
    if maximum_rank < 1:
        raise ValueError("maximum_rank must be positive")
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")
    if not torch.isfinite(samples).all():
        raise ValueError("PCA samples must be finite")

    values = samples.float()
    if center:
        values = values - values.mean(dim=0, keepdim=True)
    rows, dimension = values.shape
    total_energy = float(values.square().sum(dtype=torch.float64).item())
    if total_energy == 0.0:
        return PrincipalComponentFit(
            basis=torch.empty(dimension, 0, dtype=values.dtype, device=values.device),
            singular_values=torch.empty(0, dtype=values.dtype, device=values.device),
            effective_rank=0,
            retained_energy_fraction=0.0,
        )

    # Forming a Gram matrix squares the condition number.  Its FP32
    # eigenspectrum therefore needs an eigenvalue-domain roundoff floor;
    # comparing square roots against eps admits fake null-space directions.
    relative_eigenvalue_floor = max(
        tolerance * tolerance,
        torch.finfo(values.dtype).eps * max(rows, dimension),
    )

    if rows <= dimension:
        gram = values @ values.T
        eigenvalues, eigenvectors = torch.linalg.eigh((gram + gram.T) * 0.5)
        order = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[order].clamp_min(0)
        left = eigenvectors[:, order]
        singular = eigenvalues.sqrt()
        keep = eigenvalues > eigenvalues[0] * relative_eigenvalue_floor
        selected = min(maximum_rank, int(keep.sum().item()))
        singular = singular[:selected]
        if selected:
            basis = values.T @ left[:, :selected]
            basis = basis / singular.unsqueeze(0)
            # Keep every right-singular vector paired with its singular value.
            # A QR pass can rotate this basis and silently invalidate that
            # pairing when callers reconstruct a covariance factor.
            basis = basis / torch.linalg.vector_norm(
                basis, dim=0, keepdim=True
            ).clamp_min(torch.finfo(values.dtype).tiny)
        else:
            basis = torch.empty(
                dimension, 0, dtype=values.dtype, device=values.device
            )
    else:
        gram = values.T @ values
        eigenvalues, eigenvectors = torch.linalg.eigh((gram + gram.T) * 0.5)
        order = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[order].clamp_min(0)
        singular = eigenvalues.sqrt()
        keep = eigenvalues > eigenvalues[0] * relative_eigenvalue_floor
        selected = min(maximum_rank, int(keep.sum().item()))
        singular = singular[:selected]
        basis = eigenvectors[:, order[:selected]]

    retained = float(
        singular.square().sum(dtype=torch.float64).item() / total_energy
    )
    return PrincipalComponentFit(
        basis=basis.contiguous(),
        singular_values=singular.contiguous(),
        effective_rank=selected,
        retained_energy_fraction=min(1.0, max(0.0, retained)),
    )
