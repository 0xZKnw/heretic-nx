from __future__ import annotations

import torch

from heretic_nx.geometry.pca import exact_principal_components


def test_exact_pca_matches_direct_svd_subspace_and_spectrum() -> None:
    generator = torch.Generator().manual_seed(347)
    samples = torch.randn(96, 257, generator=generator)

    fit = exact_principal_components(samples, maximum_rank=8)
    centered = samples - samples.mean(dim=0, keepdim=True)
    _left, singular, right_t = torch.linalg.svd(centered, full_matrices=False)

    torch.testing.assert_close(fit.singular_values, singular[:8], rtol=2e-4, atol=2e-4)
    cosines = torch.linalg.svdvals(fit.basis.T @ right_t[:8].T)
    assert float(cosines.min()) > 0.9999
    assert fit.effective_rank == 8


def test_exact_pca_drops_rank_deficient_null_axes() -> None:
    axis = torch.linspace(-1.0, 1.0, 128)
    samples = torch.stack((axis, axis, -axis, -axis))

    fit = exact_principal_components(samples, maximum_rank=4)

    assert fit.effective_rank == 1
    assert fit.basis.shape == (128, 1)
    assert fit.retained_energy_fraction > 0.99999


def test_exact_pca_preserves_singular_vector_pairing() -> None:
    generator = torch.Generator().manual_seed(353)
    left = torch.randn(32, 3, generator=generator)
    right, _ = torch.linalg.qr(torch.randn(41, 3, generator=generator))
    samples = left @ torch.diag(torch.tensor([9.0, 3.0, 0.5])) @ right.T

    fit = exact_principal_components(samples, maximum_rank=3)
    centered = samples - samples.mean(dim=0, keepdim=True)
    reconstructed = (fit.basis * fit.singular_values[None, :]) @ (
        fit.basis * fit.singular_values[None, :]
    ).T

    torch.testing.assert_close(
        reconstructed,
        centered.T @ centered,
        rtol=3e-4,
        atol=3e-4,
    )


def test_exact_pca_handles_zero_variance_without_arbitrary_axes() -> None:
    fit = exact_principal_components(torch.ones(5, 12), maximum_rank=8)
    assert fit.effective_rank == 0
    assert fit.basis.shape == (12, 0)
    assert fit.singular_values.numel() == 0
    assert fit.retained_energy_fraction == 0.0
