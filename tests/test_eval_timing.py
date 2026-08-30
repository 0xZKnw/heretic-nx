from __future__ import annotations

import threading

import pytest

from heretic_nx.eval.funnel import CandidateKey, CandidateSummary, FunnelStage
from heretic_nx.eval.timing import (
    EvaluationWorkItem,
    GateSurvival,
    RunPhase,
    RunTimingProfiler,
    StageCost,
    estimate_portfolio_eta,
    execute_refusal_first_wave,
    plan_refusal_first_wave,
)


def sha(index: int) -> str:
    return f"{index:064x}"


def summary(
    index: int,
    stage: FunnelStage,
    *,
    refusal_evaluated: int = 0,
    refusal_total: int = 3,
    kl_evaluated: int = 0,
    kl_total: int = 3,
) -> CandidateSummary:
    candidate = CandidateKey(
        artifact_sha256=sha(100 + index),
        protocol_sha256=sha(200),
        geometry_split_sha256=sha(300),
        search_split_sha256=sha(400),
        public_test_split_sha256=sha(500),
    )
    return CandidateSummary(
        candidate=candidate,
        next_stage=stage,
        refusal_evaluated=refusal_evaluated,
        refusal_total=refusal_total,
        lexical_proxy_hits=0,
        semantic_refusals=0,
        kl_evaluated=kl_evaluated,
        kl_total=kl_total,
        kl_sum=0.0,
        kl_lower_bound=0.0,
        mean_kl=None,
        capability_passed=None,
        public_reported=False,
    )


def test_timing_report_separates_wall_overlap_cache_and_failures() -> None:
    profiler = RunTimingProfiler()
    profiler.record(RunPhase.REFUSAL, 4.0, units=2, started_at=0.0)
    profiler.record(
        RunPhase.REFUSAL,
        0.4,
        units=2,
        cache_hit=True,
        started_at=4.0,
    )
    profiler.record(RunPhase.KL, 2.0, started_at=1.0)
    profiler.record(
        RunPhase.REFUSAL,
        1.0,
        succeeded=False,
        started_at=5.0,
    )

    report = profiler.report()
    assert report.elapsed_wall_seconds == pytest.approx(6.0)
    assert report.active_wall_seconds == pytest.approx(5.4)
    assert report.task_seconds == pytest.approx(7.4)
    assert report.overlap_factor == pytest.approx(7.4 / 5.4)

    refusal = report.phase(RunPhase.REFUSAL)
    assert refusal is not None
    assert refusal.sample_count == 3
    assert refusal.successful_samples == 2
    assert refusal.failed_samples == 1
    assert refusal.cold_work_units == 2
    assert refusal.cache_hit_units == 2
    assert refusal.p50_seconds_per_unit == pytest.approx(2.0)
    assert refusal.p90_seconds_per_unit == pytest.approx(2.0)
    assert refusal.estimated_cache_seconds_saved == pytest.approx(3.6)


def test_measure_records_failed_attempt_without_using_it_as_a_cost() -> None:
    profiler = RunTimingProfiler()
    with pytest.raises(RuntimeError, match="boom"):
        with profiler.measure(RunPhase.KL):
            raise RuntimeError("boom")

    report = profiler.report()
    kl = report.phase(RunPhase.KL)
    assert kl is not None
    assert kl.failed_samples == 1
    assert kl.p50_seconds_per_unit is None


def test_refusal_first_wave_is_a_global_barrier() -> None:
    refusal = summary(1, FunnelStage.REFUSAL, refusal_evaluated=1)
    kl = summary(
        2,
        FunnelStage.KL,
        refusal_evaluated=3,
        kl_evaluated=1,
    )
    capability = summary(
        3,
        FunnelStage.CAPABILITY,
        refusal_evaluated=3,
        kl_evaluated=3,
    )
    complete = summary(
        4,
        FunnelStage.COMPLETE,
        refusal_evaluated=3,
        kl_evaluated=3,
    )

    wave = plan_refusal_first_wave((kl, complete, capability, refusal))
    assert wave == (
        EvaluationWorkItem(refusal.candidate.sha256, FunnelStage.REFUSAL, 2),
    )
    next_wave = plan_refusal_first_wave((kl, complete, capability))
    assert next_wave == (
        EvaluationWorkItem(kl.candidate.sha256, FunnelStage.KL, 2),
    )


def test_public_wave_requires_one_explicitly_frozen_winner() -> None:
    first = summary(
        1,
        FunnelStage.PUBLIC_REPORT,
        refusal_evaluated=3,
        kl_evaluated=3,
    )
    second = summary(
        2,
        FunnelStage.PUBLIC_REPORT,
        refusal_evaluated=3,
        kl_evaluated=3,
    )
    with pytest.raises(ValueError, match="explicitly frozen"):
        plan_refusal_first_wave((first, second))
    assert plan_refusal_first_wave(
        (first, second),
        public_candidate_sha256=second.candidate.sha256,
    ) == (
        EvaluationWorkItem(second.candidate.sha256, FunnelStage.PUBLIC_REPORT, 1),
    )


def test_refusal_first_wave_parallelizes_only_independent_same_stage_work() -> None:
    work = (
        EvaluationWorkItem(sha(1), FunnelStage.REFUSAL, 1),
        EvaluationWorkItem(sha(2), FunnelStage.REFUSAL, 1),
    )
    barrier = threading.Barrier(2, timeout=2.0)
    lock = threading.Lock()
    active = 0
    maximum_active = 0
    profiler = RunTimingProfiler()

    def worker(item: EvaluationWorkItem) -> str:
        nonlocal active, maximum_active
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
        barrier.wait()
        with lock:
            active -= 1
        return item.candidate_sha256

    results = execute_refusal_first_wave(
        work,
        worker,
        max_workers=2,
        profiler=profiler,
    )
    assert results == (sha(1), sha(2))
    assert maximum_active == 2
    refusal = profiler.report().phase(RunPhase.REFUSAL)
    assert refusal is not None
    assert refusal.sample_count == 2

    mixed = work + (EvaluationWorkItem(sha(3), FunnelStage.KL, 1),)
    with pytest.raises(ValueError, match="cannot mix"):
        execute_refusal_first_wave(mixed, worker, max_workers=1)


def test_eta_reports_critical_path_funnel_savings_and_eight_hour_speedup() -> None:
    refusal = summary(1, FunnelStage.REFUSAL, refusal_evaluated=1)
    kl = summary(
        2,
        FunnelStage.KL,
        refusal_evaluated=3,
        kl_evaluated=1,
    )
    eta = estimate_portfolio_eta(
        (refusal, kl),
        priors={
            RunPhase.REFUSAL: 2.0,
            RunPhase.KL: 5.0,
            RunPhase.CAPABILITY: 10.0,
            RunPhase.PUBLIC_REPORT: 1.0,
        },
        survival=GateSurvival(refusal=0.5, kl=0.5, capability=1.0),
        workers=2,
        baseline_seconds=80.0,
    )

    # Selection chains are 14 s and 15 s. One frozen winner gets public eval;
    # its expected shared tail is (1 - (1 - .25)*(1 - .5))*1 = .625 s.
    assert eta.p50.serial_seconds == pytest.approx(29.625)
    assert eta.p50.critical_path_seconds == pytest.approx(15.625)
    assert eta.p50.ideal_wall_seconds == pytest.approx(15.625)
    assert eta.p50.list_schedule_upper_seconds == pytest.approx(22.625)
    assert eta.naive_p50_serial_seconds == pytest.approx(50.0)
    assert eta.refusal_first_serial_speedup == pytest.approx(50.0 / 29.625)
    assert eta.baseline_speedup_lower == pytest.approx(80.0 / 22.625)
    assert eta.baseline_speedup_upper == pytest.approx(80.0 / 15.625)


def test_eta_includes_measured_geometry_and_build_prelude() -> None:
    candidate = summary(1, FunnelStage.REFUSAL)
    eta = estimate_portfolio_eta(
        (candidate,),
        priors={
            RunPhase.GEOMETRY: StageCost(10.0, 12.0),
            RunPhase.ARTIFACT_BUILD: StageCost(5.0, 7.0),
            RunPhase.REFUSAL: 1.0,
            RunPhase.KL: 2.0,
            RunPhase.CAPABILITY: 3.0,
            RunPhase.PUBLIC_REPORT: 4.0,
        },
        prelude_units={RunPhase.GEOMETRY: 1, RunPhase.ARTIFACT_BUILD: 2},
    )

    assert eta.prelude_units == (
        (RunPhase.ARTIFACT_BUILD, 2),
        (RunPhase.GEOMETRY, 1),
    )
    assert eta.p50.serial_seconds == pytest.approx(36.0)
    assert eta.p90.serial_seconds == pytest.approx(42.0)
    assert eta.p50.ideal_wall_seconds == eta.p50.serial_seconds


@pytest.mark.parametrize(
    "prelude_units, message",
    [
        ({RunPhase.KL: 1}, "only geometry and artifact-build"),
        ({RunPhase.GEOMETRY: -1}, "non-negative integer"),
        ({RunPhase.GEOMETRY: True}, "non-negative integer"),
    ],
)
def test_eta_rejects_invalid_prelude_units(
    prelude_units: dict[RunPhase, int],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        estimate_portfolio_eta((), prelude_units=prelude_units)


def test_eta_prefers_observations_and_fails_closed_on_unknown_costs() -> None:
    candidate = summary(1, FunnelStage.REFUSAL, refusal_evaluated=2)
    profiler = RunTimingProfiler()
    profiler.record(RunPhase.REFUSAL, 1.0, started_at=0.0)
    profiler.record(RunPhase.REFUSAL, 3.0, started_at=2.0)
    eta = estimate_portfolio_eta(
        (candidate,),
        profiler=profiler,
        priors={
            RunPhase.REFUSAL: StageCost(100.0, 100.0),
            RunPhase.KL: 2.0,
            RunPhase.CAPABILITY: 3.0,
            RunPhase.PUBLIC_REPORT: 4.0,
        },
    )
    costs = dict(eta.costs)
    assert costs[RunPhase.REFUSAL].source == "observed"
    assert costs[RunPhase.REFUSAL].p50_seconds == pytest.approx(2.0)
    assert costs[RunPhase.REFUSAL].p90_seconds == pytest.approx(3.0)

    with pytest.raises(ValueError, match="capability, kl, public-report"):
        estimate_portfolio_eta(
            (candidate,),
            priors={RunPhase.REFUSAL: 1.0},
        )


@pytest.mark.parametrize(
    "survival",
    [
        GateSurvival(),
        GateSurvival(refusal=0.0, kl=0.0, capability=0.0),
    ],
)
def test_single_worker_eta_equals_serial_work(survival: GateSurvival) -> None:
    candidate = summary(1, FunnelStage.REFUSAL)
    eta = estimate_portfolio_eta(
        (candidate,),
        priors={phase: 1.0 for phase in (
            RunPhase.REFUSAL,
            RunPhase.KL,
            RunPhase.CAPABILITY,
            RunPhase.PUBLIC_REPORT,
        )},
        survival=survival,
        workers=1,
    )
    assert eta.p50.ideal_wall_seconds == eta.p50.serial_seconds
    assert eta.p50.list_schedule_upper_seconds == eta.p50.serial_seconds
