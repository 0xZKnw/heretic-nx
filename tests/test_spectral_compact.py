from __future__ import annotations

import pytest
import torch

from heretic_nx.edits.spectral import fit_signed_spectral_operator


def _dense_reference(
    target: torch.Tensor,
    protected: torch.Tensor,
    *,
    protected_weight: float,
    rank: int,
    positive_only: bool,
    tolerance: float = 1e-7,
) -> tuple[torch.Tensor, torch.Tensor]:
    target = target.float()
    protected = protected.to(target)
    target = target / torch.linalg.vector_norm(target)
    protected = protected / torch.linalg.vector_norm(protected)
    contrast = target @ target.T - protected_weight * (protected @ protected.T)
    contrast = (contrast + contrast.T) / 2
    eigenvalues, eigenvectors = torch.linalg.eigh(contrast)
    order = torch.argsort(eigenvalues.abs(), descending=True, stable=True)
    if positive_only:
        order = order[eigenvalues[order] > tolerance]
    else:
        order = order[eigenvalues[order].abs() > tolerance]
    selected = order[:rank]
    return eigenvalues[selected], eigenvectors[:, selected]


@pytest.mark.parametrize("positive_only", [False, True])
def test_compact_spectral_operator_matches_dense_reference(positive_only: bool) -> None:
    generator = torch.Generator().manual_seed(307)
    target = torch.randn(47, 5, generator=generator)
    protected = torch.randn(47, 4, generator=generator)
    expected_values, expected_basis = _dense_reference(
        target,
        protected,
        protected_weight=0.65,
        rank=6,
        positive_only=positive_only,
    )

    edit = fit_signed_spectral_operator(
        target,
        protected,
        rank=6,
        beta=0.4,
        protected_weight=0.65,
        positive_only=positive_only,
    )

    torch.testing.assert_close(
        edit.contrast_eigenvalues,
        expected_values,
        atol=2e-6,
        rtol=2e-5,
    )
    overlap = torch.linalg.svdvals(expected_basis.T @ edit.basis)
    torch.testing.assert_close(overlap, torch.ones_like(overlap), atol=2e-5, rtol=2e-5)
    expected_coefficients = expected_values / expected_values.abs().max()
    expected_map = expected_basis @ torch.diag(expected_coefficients) @ expected_basis.T
    actual_map = edit.operator.a @ edit.operator.b.T
    torch.testing.assert_close(actual_map, expected_map, atol=3e-6, rtol=3e-5)


def test_compact_spectral_eigh_never_uses_ambient_square_matrix(monkeypatch) -> None:
    observed_shapes: list[tuple[int, int]] = []
    original_eigh = torch.linalg.eigh

    def recording_eigh(value, *args, **kwargs):
        observed_shapes.append(tuple(value.shape))
        return original_eigh(value, *args, **kwargs)

    monkeypatch.setattr(torch.linalg, "eigh", recording_eigh)
    generator = torch.Generator().manual_seed(311)
    edit = fit_signed_spectral_operator(
        torch.randn(2048, 8, generator=generator),
        torch.randn(2048, 8, generator=generator),
        rank=8,
        beta=0.5,
        positive_only=False,
    )

    assert edit.basis.shape == (2048, 8)
    assert observed_shapes
    assert max(shape[0] for shape in observed_shapes) <= 16


@pytest.mark.parametrize(
    ("target", "protected", "protected_weight", "message"),
    [
        (torch.zeros(12, 2), torch.ones(12, 2), 1.0, "target_factor"),
        (torch.full((12, 2), 1e-8), torch.ones(12, 2), 1.0, "target_factor"),
        (torch.ones(12, 2), torch.zeros(12, 2), 1.0, "protected_factor"),
        (torch.ones(12, 2), torch.full((12, 2), 1e-8), 1.0, "protected_factor"),
        (torch.ones(12, 0), torch.ones(12, 2), 1.0, "target_factor"),
        (torch.ones(12, 2), torch.ones(12, 0), 1.0, "protected_factor"),
    ],
)
def test_compact_spectral_rejects_degenerate_required_factors(
    target: torch.Tensor,
    protected: torch.Tensor,
    protected_weight: float,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        fit_signed_spectral_operator(
            target,
            protected,
            rank=1,
            beta=0.5,
            protected_weight=protected_weight,
        )


def test_zero_protected_weight_allows_an_empty_protected_factor() -> None:
    target = torch.randn(13, 3, generator=torch.Generator().manual_seed(313))
    edit = fit_signed_spectral_operator(
        target,
        torch.empty(13, 0),
        rank=3,
        beta=0.5,
        protected_weight=0.0,
    )
    assert edit.basis.shape == (13, 3)
    assert torch.all(edit.contrast_eigenvalues > 0)


def test_compact_spectral_rejects_cancelled_contrast() -> None:
    factor = torch.randn(19, 3, generator=torch.Generator().manual_seed(317))
    with pytest.raises(ValueError, match="no eligible direction"):
        fit_signed_spectral_operator(
            factor,
            factor.clone(),
            rank=3,
            beta=0.5,
            protected_weight=1.0,
            positive_only=False,
        )


def test_compact_spectral_supports_finite_autograd() -> None:
    generator = torch.Generator().manual_seed(331)
    target = torch.randn(23, 3, generator=generator, requires_grad=True)
    protected = torch.randn(23, 2, generator=generator, requires_grad=True)
    hidden = torch.randn(7, 23, generator=generator)
    edit = fit_signed_spectral_operator(
        target,
        protected,
        rank=3,
        beta=0.3,
        protected_weight=0.4,
        positive_only=False,
    )

    edit.operator.apply(hidden).square().mean().backward()

    assert target.grad is not None and torch.isfinite(target.grad).all()
    assert protected.grad is not None and torch.isfinite(protected.grad).all()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rank": 0, "beta": 0.5},
        {"rank": 1, "beta": float("nan")},
        {"rank": 1, "beta": 1.1},
        {"rank": 1, "beta": 0.5, "protected_weight": float("inf")},
        {"rank": 1, "beta": 0.5, "tolerance": 0.0},
    ],
)
def test_compact_spectral_rejects_invalid_parameters(kwargs: dict[str, float]) -> None:
    target = torch.ones(8, 1)
    protected = torch.eye(8)[:, 1:2]
    with pytest.raises(ValueError):
        fit_signed_spectral_operator(target, protected, **kwargs)
