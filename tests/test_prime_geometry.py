from __future__ import annotations

import torch

import heretic_nx.geometry.metric as metric_module
import heretic_nx.geometry.principal_angles as principal_angles_module
from heretic_nx.geometry.consensus import grassmann_consensus
from heretic_nx.geometry.contrastive import fit_contrastive_axis
from heretic_nx.geometry.fisher import fisher_factor_from_gradients
from heretic_nx.geometry.leace import fit_leace
from heretic_nx.geometry.metric import (
    LowRankMetric,
    MetricGeometryGate,
    metric_orthonormal_basis,
    metric_residualize,
    require_static_geometry,
)
from heretic_nx.geometry.residual import (
    fit_residual_stream_axes,
    last_token_residual_stack,
)
from heretic_nx.geometry.principal_angles import orthonormal_basis
from heretic_nx.sketches.crosscov import CrossCovarianceState


def test_streaming_crosscov_matches_direct_and_merge() -> None:
    generator = torch.Generator().manual_seed(101)
    x = torch.randn(97, 7, generator=generator, dtype=torch.float64)
    z = torch.randn(97, 3, generator=generator, dtype=torch.float64)
    full = CrossCovarianceState.empty(7, 3)
    full.update(x, z)
    left = CrossCovarianceState.empty(7, 3)
    right = CrossCovarianceState.empty(7, 3)
    left.update(x[:41], z[:41])
    right.update(x[41:], z[41:])
    left.merge(right)
    expected = (x - x.mean(0)).T @ (z - z.mean(0)) / (x.shape[0] - 1)
    torch.testing.assert_close(full.covariance, expected)
    torch.testing.assert_close(left.covariance, expected)


def test_consensus_is_invariant_to_internal_rotations() -> None:
    generator = torch.Generator().manual_seed(103)
    base = orthonormal_basis(torch.randn(20, 3, generator=generator))
    rotations = [
        orthonormal_basis(torch.randn(3, 3, generator=generator))
        for _ in range(4)
    ]
    result = grassmann_consensus(
        [base @ rotation for rotation in rotations],
        eigenvalue_minimum=0.5,
        stability_mass=0.99,
    )
    assert result.selected_rank == 3
    assert float(torch.linalg.svdvals(base.T @ result.basis).min()) > 0.999
    torch.testing.assert_close(result.eigenvalues[:3], torch.ones(3), atol=1e-5, rtol=1e-5)


def test_metric_residualization_is_m_orthogonal() -> None:
    generator = torch.Generator().manual_seed(107)
    samples = torch.randn(80, 12, generator=generator)
    fisher = fisher_factor_from_gradients(torch.randn(30, 12, generator=generator), rank=4)
    metric = LowRankMetric.from_samples(samples, fisher_factor=fisher)
    protected = torch.randn(12, 3, generator=generator)
    target = torch.randn(12, 4, generator=generator)
    editable = metric_residualize(target, protected, metric)
    protected_q = metric_orthonormal_basis(protected, metric)
    assert float((protected_q.T @ metric.apply(editable)).norm()) < 2e-4
    torch.testing.assert_close(
        editable.T @ metric.apply(editable),
        torch.eye(editable.shape[1]),
        atol=2e-4,
        rtol=2e-4,
    )
    result = MetricGeometryGate().evaluate(target, protected, metric)
    assert 0 <= result.retained_energy <= 1.0001


def test_metric_gate_reuses_precomputed_metric_bases(monkeypatch) -> None:
    generator = torch.Generator().manual_seed(108)
    metric = LowRankMetric.from_samples(
        torch.randn(40, 12, generator=generator)
    )
    target = torch.randn(12, 3, generator=generator)
    protected = torch.randn(12, 2, generator=generator)
    original = metric_module.metric_orthonormal_basis
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(metric_module, "metric_orthonormal_basis", counted)
    result = MetricGeometryGate().evaluate(target, protected, metric)

    assert calls == 3  # target, protected, and the residual; no repeated angle fit
    assert torch.isfinite(result.editable_basis).all()


def test_euclidean_gate_reuses_precomputed_bases(monkeypatch) -> None:
    generator = torch.Generator().manual_seed(110)
    target = torch.randn(32, 4, generator=generator)
    protected = torch.randn(32, 3, generator=generator)
    original = principal_angles_module.orthonormal_basis
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(principal_angles_module, "orthonormal_basis", counted)
    result = principal_angles_module.GeometryGate().evaluate(target, protected)

    assert calls == 3  # target, protected, and projected residual
    assert torch.isfinite(result.editable_basis).all()


def test_leace_removes_linear_cross_covariance() -> None:
    generator = torch.Generator().manual_seed(109)
    labels = torch.randint(0, 3, (600,), generator=generator)
    concepts = torch.nn.functional.one_hot(labels, num_classes=3).float()
    signal = concepts @ torch.randn(3, 10, generator=generator)
    values = signal + 0.3 * torch.randn(600, 10, generator=generator)
    eraser = fit_leace(values, concepts)
    erased = eraser.apply(values)
    cross = (erased - erased.mean(0)).T @ (concepts - concepts.mean(0)) / values.shape[0]
    assert float(cross.norm()) < 2e-4
    assert eraser.concept_rank == 2


def test_rejected_metric_geometry_cannot_be_used_for_static_edit() -> None:
    metric = LowRankMetric.from_factors(6)
    protected = torch.eye(6)[:, :2]
    result = MetricGeometryGate().evaluate(protected[:, :1], protected, metric)
    assert result.decision == "reject-site"
    try:
        require_static_geometry(result, site_id="L0:block")
    except RuntimeError as error:
        assert "not eligible" in str(error)
    else:
        raise AssertionError("a rejected gate must never yield a static editor")


def test_metric_rejects_non_finite_and_mixed_device_inputs() -> None:
    try:
        LowRankMetric.from_samples(torch.tensor([[0.0, 1.0], [float("nan"), 2.0]]))
    except ValueError as error:
        assert "finite" in str(error)
    else:
        raise AssertionError("non-finite metric calibration must fail")


def test_contrastive_axis_is_stable_and_safe_mean_orthogonal() -> None:
    generator = torch.Generator().manual_seed(113)
    safe_mean = torch.tensor([2.0, 0.0, 0.0, 0.0])
    target_axis = torch.tensor([0.0, 0.0, 1.0, 0.0])
    safe = safe_mean + 0.05 * torch.randn(90, 4, generator=generator)
    target = safe_mean + target_axis + 0.05 * torch.randn(90, 4, generator=generator)
    evidence = fit_contrastive_axis(safe, target, folds=3)
    assert float(torch.dot(evidence.axis, target_axis)) > 0.99
    assert abs(evidence.safe_mean_cosine) < 1e-5
    assert evidence.fold_cosine_minimum > 0.99


def test_last_token_residual_stack_supports_left_and_right_padding() -> None:
    embedding = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
    layer_0 = embedding + 100
    layer_1 = embedding + 200
    left_mask = torch.tensor([[0, 0, 1, 1], [0, 1, 1, 1]])
    right_mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]])
    left = last_token_residual_stack((embedding, layer_0, layer_1), left_mask)
    right = last_token_residual_stack((embedding, layer_0, layer_1), right_mask)
    torch.testing.assert_close(left[:, 0], layer_0[:, 3])
    torch.testing.assert_close(right[:, 0], torch.stack((layer_0[0, 1], layer_0[1, 2])))
    assert left.shape == (2, 2, 3)


def test_residual_stream_axes_are_layer_aligned() -> None:
    generator = torch.Generator().manual_seed(127)
    safe = 0.02 * torch.randn(90, 2, 5, generator=generator)
    target = safe.clone()
    target[:, 0, 1] += 1
    target[:, 1, 3] += 1
    axes = fit_residual_stream_axes(
        safe, target, folds=3, remove_safe_mean=False
    )
    assert len(axes) == 2
    assert float(axes[0].axis[1]) > 0.99
    assert float(axes[1].axis[3]) > 0.99
