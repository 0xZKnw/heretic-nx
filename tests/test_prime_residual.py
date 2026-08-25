import pytest
import torch

from heretic_nx.geometry.residual import (
    fit_residual_stream_axes,
    protect_residual_stream_axes,
)


def test_capability_protected_residual_axes_are_normalized_and_orthogonal() -> None:
    generator = torch.Generator().manual_seed(17)
    safe = torch.randn(36, 2, 12, generator=generator)
    safe[:, :, :2] *= 8.0
    target = safe.clone()
    target[:, :, 2] += 3.0
    axes = fit_residual_stream_axes(safe, target, folds=3)

    protected = protect_residual_stream_axes(
        safe,
        target,
        axes,
        capability_rank=2,
        oversample=2,
        seed=91,
    )

    assert len(protected) == 2
    for result in protected:
        axis = result.evidence.axis
        assert torch.linalg.vector_norm(axis).item() == pytest.approx(1.0, abs=1e-5)
        assert torch.linalg.vector_norm(result.capability_basis.T @ axis).item() < 1e-5
        assert result.retained_fraction > 0.0
        assert result.safe_projection_rms > 0.0
        assert result.target_separation > 0.0
        assert result.efficiency > 0.0


def test_capability_protection_requires_one_axis_per_layer() -> None:
    values = torch.randn(12, 2, 5)
    axes = fit_residual_stream_axes(values, values + 0.1, folds=3)

    with pytest.raises(ValueError, match="one contrastive axis"):
        protect_residual_stream_axes(values, values + 0.1, axes[:1])
