from __future__ import annotations

import torch

from heretic_nx.edits.cayley import apply_cayley, cayley_matrix
from heretic_nx.edits.projector import apply_projector, projector_factors
from heretic_nx.geometry.principal_angles import GeometryGate, orthonormal_basis
from heretic_nx.optimize.rank_allocator import allocate_rank
from heretic_nx.optimize.waterfill import solve_kkt
from heretic_nx.sketches.frequent_directions import FrequentDirections
from heretic_nx.sketches.welford import WelfordState


def test_welford_matches_direct_and_merge() -> None:
    generator = torch.Generator().manual_seed(17)
    values = torch.randn(101, 12, generator=generator, dtype=torch.float64)
    direct = WelfordState.empty(12)
    direct.update(values)
    left = WelfordState.empty(12)
    right = WelfordState.empty(12)
    left.update(values[:47])
    right.update(values[47:])
    left.merge(right)
    torch.testing.assert_close(direct.mean, values.mean(0))
    torch.testing.assert_close(direct.variance, values.var(0))
    torch.testing.assert_close(left.mean, direct.mean)
    torch.testing.assert_close(left.variance, direct.variance)


def test_frequent_directions_recovers_dominant_subspace() -> None:
    generator = torch.Generator().manual_seed(29)
    latent = torch.randn(400, 3, generator=generator)
    mixing = orthonormal_basis(torch.randn(16, 3, generator=generator))
    values = latent @ mixing.T + 0.01 * torch.randn(400, 16, generator=generator)
    sketch = FrequentDirections(rank=6, dimension=16)
    for chunk in values.split(17):
        sketch.update(chunk)
    recovered = sketch.basis(rank=3)
    overlap = torch.linalg.svdvals(mixing.T @ recovered).min()
    assert float(overlap) > 0.99
    assert sketch.rows.shape[0] <= 6


def test_geometry_gate_rejects_protected_target() -> None:
    protected = torch.eye(8)[:, :2]
    rejected = GeometryGate().evaluate(protected[:, :1], protected)
    accepted = GeometryGate().evaluate(torch.eye(8)[:, 4:6], protected)
    assert rejected.decision == "reject-site"
    assert accepted.decision == "safe-static"
    torch.testing.assert_close(
        accepted.editable_basis.T @ accepted.editable_basis,
        torch.eye(2),
    )


def test_projector_matches_lora_factorization() -> None:
    generator = torch.Generator().manual_seed(43)
    weight = torch.randn(10, 7, generator=generator)
    q = orthonormal_basis(torch.randn(10, 3, generator=generator))
    beta = 0.31
    factors = projector_factors(weight, q, beta)
    expected = (torch.eye(10) - beta * q @ q.T) @ weight
    torch.testing.assert_close(factors.apply_weight(weight), expected)
    outputs = torch.randn(5, 10, generator=generator)
    torch.testing.assert_close(apply_projector(outputs, q, beta), outputs @ (torch.eye(10) - beta * q @ q.T))


def test_cayley_is_orthogonal_and_preserves_norms() -> None:
    generator = torch.Generator().manual_seed(59)
    u = 0.1 * torch.randn(12, 2, generator=generator, dtype=torch.float64)
    v = 0.1 * torch.randn(12, 2, generator=generator, dtype=torch.float64)
    rotation = cayley_matrix(u, v)
    torch.testing.assert_close(rotation.T @ rotation, torch.eye(12, dtype=torch.float64), atol=1e-10, rtol=1e-10)
    values = torch.randn(9, 12, generator=generator, dtype=torch.float64)
    rotated = apply_cayley(values, u, v)
    torch.testing.assert_close(rotated.norm(dim=-1), values.norm(dim=-1), atol=1e-10, rtol=1e-10)


def test_kkt_respects_budget_and_bounds() -> None:
    gain = torch.tensor([1.2, 0.7, 0.2], dtype=torch.float64)
    curvature = torch.tensor([1.0, 2.0, 1.0], dtype=torch.float64)
    cost = torch.tensor([1.0, 1.5, 0.5], dtype=torch.float64)
    result = solve_kkt(gain, curvature, cost, budget=0.5, beta_max=0.8)
    assert result.spent_budget <= 0.5 + 1e-6
    assert torch.all(result.beta >= 0)
    assert torch.all(result.beta <= 0.8)


def test_rank_allocator_obeys_budget_and_prefixes() -> None:
    result = allocate_rank(
        {"a": [5.0, 2.0], "b": [4.0, 3.0]},
        {"a": [2.0, 2.0], "b": [1.0, 3.0]},
        budget=4.0,
    )
    assert result == {"a": 1, "b": 1}


def test_rank_allocator_finds_global_prefix_optimum() -> None:
    result = allocate_rank(
        {"unlock": [1.0, 100.0], "greedy-trap": [10.0]},
        {"unlock": [4.0, 1.0], "greedy-trap": [5.0]},
        budget=5.0,
    )
    assert result == {"greedy-trap": 0, "unlock": 2}


def test_frequent_directions_is_centered_translation_invariant_and_mergeable() -> None:
    generator = torch.Generator().manual_seed(157)
    values = torch.randn(300, 10, generator=generator)
    shifted = values + torch.arange(10) * 100.0
    original = FrequentDirections(rank=6, dimension=10)
    translated = FrequentDirections(rank=6, dimension=10)
    left = FrequentDirections(rank=6, dimension=10)
    right = FrequentDirections(rank=6, dimension=10)
    original.update(values)
    translated.update(shifted)
    left.update(values[:137])
    right.update(values[137:])
    left.merge(right)
    original_basis = original.basis(4)
    translated_basis = translated.basis(4)
    merged_basis = left.basis(3)
    assert float(torch.linalg.svdvals(original_basis.T @ translated_basis).min()) > 0.999
    assert float(torch.linalg.svdvals(original_basis.T @ merged_basis).min()) > 0.98
    torch.testing.assert_close(translated.mean, shifted.mean(0), atol=1e-4, rtol=1e-5)
