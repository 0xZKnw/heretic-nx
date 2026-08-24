from __future__ import annotations

import torch

from heretic_nx.geometry.task_conditioned import fit_task_conditioned_geometry
from heretic_nx.geometry.token_positions import instruction_index_from_offsets
from heretic_nx.optimize.causal_scan import dual_representation_loss, symmetric_estimate
from heretic_nx.runtime.latent_router import ConsensusSafetyRouter, LatentSafetyRouter


def test_instruction_position_ignores_template_suffix() -> None:
    offsets = [(0, 0), (0, 4), (5, 8), (8, 9), (0, 0), (0, 0)]
    assert instruction_index_from_offsets(offsets, 0, 9) == 3


def test_task_conditioning_cancels_task_identity() -> None:
    # The task offset is much larger than the within-task refusal signal.
    rows = torch.tensor(
        [
            [10.0, 0.0, 1.0],
            [10.1, 0.0, 1.0],
            [10.0, 0.0, 0.0],
            [9.9, 0.0, 0.0],
            [0.0, -12.0, 1.0],
            [0.0, -12.1, 1.0],
            [0.0, -12.0, 0.0],
            [0.0, -11.9, 0.0],
        ]
    )
    geometry = fit_task_conditioned_geometry(
        rows,
        ["a"] * 4 + ["b"] * 4,
        [True, True, False, False] * 2,
    )
    assert len(geometry.contrasts) == 2
    refusal_axis = torch.tensor([0.0, 0.0, 1.0])
    overlap = torch.linalg.vector_norm(geometry.pooled_basis.T @ refusal_axis)
    assert float(overlap) > 0.99


def test_router_blocks_calibrated_unsafe_and_routes_safe_tasks() -> None:
    safe = torch.tensor(
        [
            [-2.0, 2.0, 0.0],
            [-1.8, 2.1, 0.1],
            [-2.0, -2.0, 0.0],
            [-2.2, -1.9, -0.1],
        ]
    )
    unsafe = torch.tensor([[2.0, 0.0, 0.0], [2.2, 0.1, 0.1], [1.8, -0.1, 0.0]])
    router = LatentSafetyRouter.fit(
        safe,
        unsafe,
        ["translate", "translate", "code", "code"],
        minimum_task_similarity=0.0,
    )
    assert all(router.decide(row).action == "abstain-harmfulness" for row in unsafe)
    assert router.decide(safe[0]).task == "translate"
    assert router.decide(safe[-1]).task == "code"

    second = LatentSafetyRouter.fit(
        safe * torch.tensor([1.0, 0.9, 1.1]),
        unsafe * torch.tensor([1.0, 0.9, 1.1]),
        ["translate", "translate", "code", "code"],
        minimum_task_similarity=0.0,
    )
    consensus = ConsensusSafetyRouter({"liv": router, "gqa": second}, task_site_id="gqa")
    assert consensus.decide({"liv": unsafe[0], "gqa": safe[0]}).action == "abstain-harmfulness"
    assert consensus.decide({"liv": safe[0], "gqa": safe[0]}).action == "route"
    assert consensus.decide({"liv": safe[0]}).action == "abstain-task"


def test_causal_estimate_and_dual_loss() -> None:
    eps = 0.1
    minus = torch.tensor([(2.0 - eps) ** 2])
    base = torch.tensor([4.0])
    plus = torch.tensor([(2.0 + eps) ** 2])
    estimate = symmetric_estimate(minus, base, plus, eps)
    torch.testing.assert_close(estimate.gradient, torch.tensor([4.0]))
    torch.testing.assert_close(estimate.curvature, torch.tensor([2.0]), atol=1e-4, rtol=1e-4)

    loss = dual_representation_loss(
        torch.ones(2),
        torch.zeros(2),
        torch.ones(2),
        torch.zeros(2),
        unsafe_weight=2.0,
    )
    torch.testing.assert_close(loss, torch.tensor(3.0))
