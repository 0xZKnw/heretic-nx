from __future__ import annotations

import pytest
import torch
from torch import Tensor

from heretic_nx.edits.affine import affine_operator_from_leace
from heretic_nx.geometry.leace import LeaceEraser, fit_leace
from heretic_nx.geometry.principal_angles import orthonormal_basis


def _dense_reference(
    representations: Tensor,
    concepts: Tensor,
    *,
    tolerance: float = 1e-7,
) -> tuple[Tensor, Tensor, int]:
    x = representations.float()
    z = concepts.float()
    mean_x = x.mean(dim=0)
    centered_x = x - mean_x
    centered_z = z - z.mean(dim=0)
    covariance = centered_x.T @ centered_x / x.shape[0]
    cross_covariance = centered_x.T @ centered_z / x.shape[0]
    eigenvalues, eigenvectors = torch.linalg.eigh((covariance + covariance.T) / 2)
    threshold = tolerance * eigenvalues.max().clamp_min(torch.finfo(x.dtype).eps)
    keep = eigenvalues > threshold
    support = eigenvectors[:, keep]
    roots = eigenvalues[keep].sqrt()
    covariance_sqrt = (support * roots) @ support.T
    whitening = (support / roots) @ support.T
    concept_basis = orthonormal_basis(
        whitening @ cross_covariance, tolerance=tolerance
    )
    if concept_basis.shape[1] == 0:
        projection = torch.eye(x.shape[1], dtype=x.dtype, device=x.device)
    else:
        projection = (
            torch.eye(x.shape[1], dtype=x.dtype, device=x.device)
            - covariance_sqrt @ concept_basis @ concept_basis.T @ whitening
        )
    bias = mean_x - projection @ mean_x
    return projection, bias, concept_basis.shape[1]


def test_thin_fit_matches_dense_reference_at_full_feature_rank() -> None:
    generator = torch.Generator().manual_seed(211)
    values = torch.randn(320, 24, generator=generator)
    concepts = torch.randn(320, 4, generator=generator)

    eraser = fit_leace(values, concepts)
    expected_projection, expected_bias, expected_rank = _dense_reference(
        values, concepts
    )

    assert eraser.concept_rank == expected_rank
    torch.testing.assert_close(
        eraser.projection, expected_projection, atol=4e-6, rtol=4e-6
    )
    torch.testing.assert_close(eraser.bias, expected_bias, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(
        eraser.apply(values),
        values @ expected_projection.T + expected_bias,
        atol=8e-6,
        rtol=4e-6,
    )


def test_rank_deficient_fit_stays_low_rank_until_projection_is_requested() -> None:
    generator = torch.Generator().manual_seed(223)
    labels = torch.randint(0, 3, (48,), generator=generator)
    concepts = torch.nn.functional.one_hot(labels, num_classes=3).float()
    values = concepts @ torch.randn(3, 512, generator=generator)
    values += 0.1 * torch.randn(48, 512, generator=generator)

    eraser = fit_leace(values, concepts)

    assert eraser._projection is None
    assert eraser.erase_left is not None
    assert eraser.erase_right is not None
    assert eraser.erase_left.shape == (512, 2)
    assert eraser.erase_right.shape == (512, 2)
    erased = eraser.apply(values)
    assert eraser._projection is None
    cross_covariance = (
        (erased - erased.mean(0)).T @ (concepts - concepts.mean(0)) / values.shape[0]
    )
    assert float(cross_covariance.norm()) < 3e-4

    projection = eraser.projection
    assert eraser._projection is projection
    torch.testing.assert_close(
        erased, values @ projection.T + eraser.bias, atol=2e-5, rtol=2e-5
    )


def test_affine_adapter_reuses_exact_factors_without_dense_materialization() -> None:
    generator = torch.Generator().manual_seed(227)
    labels = torch.randint(0, 2, (96,), generator=generator)
    concepts = torch.nn.functional.one_hot(labels, num_classes=2).float()
    values = concepts @ torch.randn(2, 256, generator=generator)
    values += 0.2 * torch.randn(96, 256, generator=generator)
    eraser = fit_leace(values, concepts)

    operator = affine_operator_from_leace(eraser)

    assert eraser._projection is None
    assert operator.linear.rank == eraser.concept_rank == 1
    torch.testing.assert_close(
        operator.apply(values), eraser.apply(values), atol=2e-5, rtol=2e-5
    )


def test_historical_dense_constructor_and_apply_remain_supported() -> None:
    projection = torch.diag(torch.tensor([1.0, 0.0, 0.5]))
    bias = torch.tensor([0.25, -0.5, 1.0])
    eraser = LeaceEraser(projection, bias, 2)
    values = torch.tensor([[2.0, 3.0, 4.0]])

    assert eraser.projection is projection
    assert eraser.erase_left is None
    assert eraser.dimension == 3
    torch.testing.assert_close(eraser.apply(values), values @ projection.T + bias)


def test_degenerate_covariance_produces_identity_map() -> None:
    values = torch.full((16, 20), 3.5)
    concepts = torch.randn(16, 3, generator=torch.Generator().manual_seed(229))

    eraser = fit_leace(values, concepts)

    assert eraser.concept_rank == 0
    assert eraser.erase_left is not None and eraser.erase_left.shape == (20, 0)
    assert eraser.erase_right is not None and eraser.erase_right.shape == (20, 0)
    torch.testing.assert_close(eraser.apply(values), values)


def test_thin_fit_remains_differentiable() -> None:
    generator = torch.Generator().manual_seed(233)
    values = torch.randn(80, 12, generator=generator, requires_grad=True)
    concepts = torch.randn(80, 2, generator=generator)

    eraser = fit_leace(values, concepts)
    loss = eraser.apply(values).square().mean()
    loss.backward()

    assert values.grad is not None
    assert torch.isfinite(values.grad).all()


def test_lazy_projection_does_not_cache_a_no_grad_detach() -> None:
    generator = torch.Generator().manual_seed(239)
    values = torch.randn(80, 12, generator=generator, requires_grad=True)
    concepts = torch.randn(80, 2, generator=generator)
    eraser = fit_leace(values, concepts)

    with torch.no_grad():
        _ = eraser.projection
    assert eraser._projection is None

    loss = eraser.projection.square().mean()
    loss.backward()
    assert values.grad is not None
    assert torch.isfinite(values.grad).all()


@pytest.mark.parametrize("tolerance", [0.0, -1.0, float("nan"), float("inf")])
def test_thin_fit_rejects_invalid_tolerance(tolerance: float) -> None:
    with pytest.raises(ValueError, match="tolerance"):
        fit_leace(torch.ones(4, 3), torch.ones(4, 1), tolerance=tolerance)


def test_thin_fit_rejects_nonfinite_or_empty_representations() -> None:
    with pytest.raises(ValueError, match="finite"):
        fit_leace(
            torch.tensor([[1.0, float("nan")], [2.0, 3.0]]),
            torch.ones(2, 1),
        )
    with pytest.raises(ValueError, match="at least one feature"):
        fit_leace(torch.empty(3, 0), torch.ones(3, 1))
