from __future__ import annotations

import torch

from heretic_nx.runtime.temporal_logits import TemporalThinkController


def _controller(*, risk: bool = True, task: bool = True) -> TemporalThinkController:
    return TemporalThinkController(
        prompt_length=2,
        open_token_id=7,
        close_token_id=8,
        budget_tokens=3,
        grace_tokens=2,
        close_logit_boost=5.0,
        risk_gate_passed=risk,
        task_route_passed=task,
    )


def test_temporal_controller_is_fail_closed() -> None:
    input_ids = torch.tensor([[1, 2, 7, 4, 5]])
    scores = torch.zeros(1, 10)
    for controller in (_controller(risk=False), _controller(task=False)):
        torch.testing.assert_close(controller(input_ids, scores), scores)
        assert controller.last_decision.active_rows == ()


def test_temporal_controller_boosts_then_forces_only_an_open_span() -> None:
    controller = _controller()
    at_budget = torch.tensor([[1, 2, 7, 4, 5]])
    scores = torch.zeros(1, 10)
    boosted = controller(at_budget, scores)
    assert float(boosted[0, 8]) == 5.0
    assert not controller.last_decision.forced

    after_grace = torch.tensor([[1, 2, 7, 4, 5, 6]])
    forced = controller(after_grace, scores)
    assert torch.isneginf(forced[0]).sum().item() == 9
    assert float(forced[0, 8]) == 0.0
    assert controller.last_decision.forced

    already_closed = torch.tensor([[1, 2, 7, 4, 8, 6, 6]])
    torch.testing.assert_close(controller(already_closed, scores), scores)
    assert controller.last_decision.active_rows == ()
