from __future__ import annotations

import random

from heretic_nx.eval.cache import JudgeCache, judge_cache_key
from heretic_nx.eval.cascade import JudgeCascade, JudgeVerdict, surface_verdict
from heretic_nx.eval.metrics import TaskOutcome, aggregate_benign_metrics
from heretic_nx.eval.sequential import (
    AnytimeBernoulliCS,
    ConfidenceInterval,
    clopper_pearson_zero_upper,
    sequential_decision,
)
from heretic_nx.runtime.token_budget import OperationBudgetRegistry, cuda_memory_caps


def test_surface_judge_abstains_on_gamed_markers() -> None:
    assert surface_verdict("As an AI, the answer is four.").label == "ambiguous"
    assert surface_verdict("I'm sorry you had to wait. Here is the result.").label == "ambiguous"
    assert surface_verdict("I cannot assist with that.").label == "refusal"
    assert surface_verdict("anything", task_success=True).label == "compliance"


def test_judge_cascade_and_cache(tmp_path) -> None:
    calls = {"j1": 0}

    def j1(_prompt: str, _response: str, _rubric: str) -> JudgeVerdict:
        calls["j1"] += 1
        return JudgeVerdict("compliance", 0.95, "J1", rationale="local head")

    with JudgeCache(tmp_path / "judge.sqlite") as cache:
        cascade = JudgeCascade(rubric="v1", j1=j1, cache=cache)
        first = cascade.judge("q", "As an AI, here is the answer")
        second = cascade.judge("q", "As an AI, here is the answer")
        assert first.label == "compliance"
        assert second.cached
        assert calls["j1"] == 1
        assert cache.get(judge_cache_key("q", "As an AI, here is the answer", "v1"))


def test_benign_success_is_primary_not_non_refusal() -> None:
    metrics = aggregate_benign_metrics(
        [
            TaskOutcome("math", True, False, False, 1.0),
            TaskOutcome("math", False, False, True, 0.0),
            TaskOutcome("code", False, True, False, 0.0),
        ]
    )
    assert metrics.benign_success_rate == 1 / 3
    assert metrics.benign_refusal_rate == 1 / 3
    assert metrics.deflection_rate == 1 / 3
    assert metrics.by_task["math"].benign_success_rate == 0.5


def test_anytime_cs_monte_carlo_coverage_and_decisions() -> None:
    generator = random.Random(149)
    misses = 0
    trials = 300
    truth = 0.3
    for _ in range(trials):
        sequence = AnytimeBernoulliCS(alpha=0.05)
        covered_all_times = True
        for _step in range(80):
            interval = sequence.update(generator.random() < truth)
            covered_all_times &= interval.lower <= truth <= interval.upper
        misses += int(not covered_all_times)
    assert misses / trials <= 0.05
    assert abs(clopper_pearson_zero_upper(2995) - 0.001) < 2e-6
    candidate = ConfidenceInterval(0.7, 0.8, 100, 0.75)
    incumbent = ConfidenceInterval(0.4, 0.5, 100, 0.45)
    assert sequential_decision(candidate, incumbent=incumbent).action == "promote"


def test_operation_budgets_are_independent() -> None:
    registry = OperationBudgetRegistry.defaults()
    before = registry["generate"].batch_controller.batch_size
    registry["backward"].batch_controller.on_oom()
    assert registry["generate"].batch_controller.batch_size == before
    assert registry["prefill"].batch_for(2048) == 2
    soft, hard = cuda_memory_caps(8 * 1024**3)
    assert soft == int(6.24 * 1024**3) or soft == int(6.25 * 1024**3)
    assert hard == int(7.2 * 1024**3)
