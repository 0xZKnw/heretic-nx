from __future__ import annotations

from heretic_nx.benchmark.closed_track import (
    ArmObservations,
    ArmRegistration,
    ClosedTrackRegistration,
    OptimizationBudget,
    evaluate_closed_track,
)


def _budget() -> OptimizationBudget:
    return OptimizationBudget(
        maximum_trials=20,
        maximum_forward_tokens=1_000_000,
        maximum_backward_tokens=100_000,
        maximum_gpu_seconds=3600,
        maximum_peak_vram_bytes=8_000_000_000,
    )


def _arm(arm_id: str, role: str) -> ArmRegistration:
    return ArmRegistration(
        id=arm_id,
        role=role,
        engine_id=f"engine/{arm_id}",
        engine_revision="abc",
        engine_config_sha256="1" * 64,
        base_model_sha256="2" * 64,
        training_data_sha256="3" * 64,
        split_manifest_sha256="4" * 64,
        budget=_budget(),
        seeds=(17, 29, 43),
    )


def _registration() -> ClosedTrackRegistration:
    return ClosedTrackRegistration(
        track_id="lfm25-closed-v1",
        benchmark_data_sha256="5" * 64,
        judge_rubric_sha256="6" * 64,
        capability_margin=0.02,
        risk_margin=0.01,
        target_superiority_margin=0.01,
        alpha=0.05,
        resamples=1000,
        arms=(_arm("nx", "candidate"), _arm("heretic", "competitor")),
    )


def _observations(arm_id: str, target: float) -> ArmObservations:
    return ArmObservations(
        arm_id=arm_id,
        output_model_sha256="7" * 64,
        response_artifact_sha256="8" * 64,
        target_scores={f"t-{index}": target for index in range(40)},
        capability_scores={f"c-{index}": 0.8 for index in range(40)},
        risk_scores={f"r-{index}": 0.01 for index in range(40)},
        trials_used=20,
        forward_tokens_used=900_000,
        backward_tokens_used=90_000,
        gpu_seconds_used=3000,
        peak_vram_bytes=7_000_000_000,
    )


def test_closed_track_requires_matched_budget_data_and_paired_superiority() -> None:
    registration = _registration()
    result = evaluate_closed_track(
        registration,
        {
            "nx": _observations("nx", 0.8),
            "heretic": _observations("heretic", 0.5),
        },
        seed=191,
    )
    assert result.benchmark_winner
    assert result.comparisons[0].target_superiority_passed
    assert result.comparisons[0].capability_noninferiority_passed
    assert result.comparisons[0].risk_noninferiority_passed


def test_closed_track_rejects_unmatched_registration_and_budget_overrun() -> None:
    competitor = _arm("heretic", "competitor").model_copy(
        update={"training_data_sha256": "9" * 64}
    )
    try:
        ClosedTrackRegistration(
            track_id="bad",
            benchmark_data_sha256="5" * 64,
            judge_rubric_sha256="6" * 64,
            capability_margin=0.02,
            risk_margin=0.01,
            target_superiority_margin=0.0,
            alpha=0.05,
            resamples=1000,
            arms=(_arm("nx", "candidate"), competitor),
        )
    except ValueError as error:
        assert "not budget/data matched" in str(error)
    else:
        raise AssertionError("unmatched engines must not share a closed track")

    registration = _registration()
    over_budget = _observations("nx", 0.8).model_copy(update={"trials_used": 21})
    result = evaluate_closed_track(
        registration,
        {"nx": over_budget, "heretic": _observations("heretic", 0.5)},
    )
    assert not result.benchmark_winner
    assert "budget:nx:trials" in result.blocking_reasons


def test_closed_track_does_not_award_a_tie() -> None:
    result = evaluate_closed_track(
        _registration(),
        {
            "nx": _observations("nx", 0.5),
            "heretic": _observations("heretic", 0.5),
        },
    )
    assert not result.benchmark_winner
    assert result.blocking_reasons == ("comparison:heretic",)
