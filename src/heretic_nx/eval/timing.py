"""Measured timing and refusal-first portfolio orchestration.

This module does not evaluate a model and never opens a split.  It provides two
small pieces of infrastructure around :class:`EvaluationFunnel`:

* a thread-safe timing ledger with honest wall, active, task, cache, and
  per-phase measurements;
* a refusal-first work planner and an ETA envelope derived from measured costs.

The ETA is deliberately an interval.  Its lower edge assumes ideal scheduling;
its upper edge is the standard work-conserving list-schedule bound for the
modeled work.  Gate survival remains an explicit assumption, and P90 means a
per-unit P90 cost envelope rather than a statistical P90 of the whole run.
Unknown phase costs fail closed instead of silently becoming zero.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
import math
import statistics
import threading
import time
from typing import TypeVar

from .funnel import CandidateSummary, FunnelStage


class RunPhase(StrEnum):
    """Stable names for end-to-end Heretic NX timing boundaries."""

    GEOMETRY = "geometry"
    ARTIFACT_BUILD = "artifact-build"
    REFUSAL = "refusal"
    KL = "kl"
    CAPABILITY = "capability"
    PUBLIC_REPORT = "public-report"


_FUNNEL_TO_RUN_PHASE = {
    FunnelStage.REFUSAL: RunPhase.REFUSAL,
    FunnelStage.KL: RunPhase.KL,
    FunnelStage.CAPABILITY: RunPhase.CAPABILITY,
    FunnelStage.PUBLIC_REPORT: RunPhase.PUBLIC_REPORT,
}
_EVALUATION_PHASES = tuple(_FUNNEL_TO_RUN_PHASE.values())


def _run_phase(value: RunPhase | FunnelStage | str) -> RunPhase:
    if isinstance(value, FunnelStage):
        try:
            return _FUNNEL_TO_RUN_PHASE[value]
        except KeyError as error:
            raise ValueError(f"{value.value!r} is not a measurable work stage") from error
    try:
        return RunPhase(value)
    except ValueError as error:
        raise ValueError(f"unsupported run phase: {value!r}") from error


def _nonnegative_finite(value: float, *, field_name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValueError(f"{field_name} must be a finite non-negative number")
    return float(value)


def _positive_units(value: int, *, field_name: str = "units") -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _nonnegative_units(value: int, *, field_name: str = "units") -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def _nearest_rank(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("cannot estimate a quantile without observations")
    ordered = sorted(values)
    rank = max(1, math.ceil(quantile * len(ordered)))
    return ordered[rank - 1]


@dataclass(frozen=True, slots=True)
class TimingSample:
    phase: RunPhase
    candidate_sha256: str | None
    units: int
    started_at: float
    elapsed_seconds: float
    cache_hit: bool
    succeeded: bool

    @property
    def ended_at(self) -> float:
        return self.started_at + self.elapsed_seconds

    @property
    def seconds_per_unit(self) -> float:
        return self.elapsed_seconds / self.units


@dataclass(frozen=True, slots=True)
class PhaseTimingStats:
    phase: RunPhase
    sample_count: int
    successful_samples: int
    failed_samples: int
    cold_work_units: int
    cache_hit_units: int
    elapsed_seconds: float
    p50_seconds_per_unit: float | None
    p90_seconds_per_unit: float | None
    estimated_cache_seconds_saved: float


@dataclass(frozen=True, slots=True)
class RunTimingReport:
    """Observed timing boundaries.

    ``elapsed_wall_seconds`` includes idle gaps between the first and last
    sample. ``active_wall_seconds`` is the union of measured intervals.
    ``task_seconds`` is their sum, so ``overlap_factor`` is greater than one
    only when work actually overlapped.
    """

    elapsed_wall_seconds: float
    active_wall_seconds: float
    task_seconds: float
    overlap_factor: float
    phases: tuple[PhaseTimingStats, ...]

    def phase(self, phase: RunPhase | FunnelStage | str) -> PhaseTimingStats | None:
        resolved = _run_phase(phase)
        return next((item for item in self.phases if item.phase is resolved), None)


class RunTimingProfiler:
    """Thread-safe phase profiler suitable for concurrent candidate workers."""

    def __init__(self, *, clock: Callable[[], float] = time.perf_counter) -> None:
        self._clock = clock
        self._samples: list[TimingSample] = []
        self._lock = threading.Lock()

    @contextmanager
    def measure(
        self,
        phase: RunPhase | FunnelStage | str,
        *,
        candidate_sha256: str | None = None,
        units: int = 1,
        cache_hit: bool = False,
    ) -> Iterator[None]:
        """Measure a phase, retaining failed attempts without forecasting from them."""

        resolved = _run_phase(phase)
        checked_units = _positive_units(units)
        if candidate_sha256 is not None and not candidate_sha256.strip():
            raise ValueError("candidate_sha256 cannot be empty")
        if not isinstance(cache_hit, bool):
            raise ValueError("cache_hit must be boolean")
        started_at = self._clock()
        succeeded = False
        try:
            yield
            succeeded = True
        finally:
            elapsed = max(0.0, self._clock() - started_at)
            self._append(
                TimingSample(
                    phase=resolved,
                    candidate_sha256=candidate_sha256,
                    units=checked_units,
                    started_at=started_at,
                    elapsed_seconds=elapsed,
                    cache_hit=cache_hit,
                    succeeded=succeeded,
                )
            )

    def record(
        self,
        phase: RunPhase | FunnelStage | str,
        elapsed_seconds: float,
        *,
        candidate_sha256: str | None = None,
        units: int = 1,
        cache_hit: bool = False,
        succeeded: bool = True,
        started_at: float | None = None,
    ) -> TimingSample:
        """Record an already measured interval.

        ``started_at`` uses the caller's monotonic clock domain.  Omitting it
        anchors the interval at the profiler clock's current value.
        """

        resolved = _run_phase(phase)
        elapsed = _nonnegative_finite(elapsed_seconds, field_name="elapsed_seconds")
        checked_units = _positive_units(units)
        if candidate_sha256 is not None and not candidate_sha256.strip():
            raise ValueError("candidate_sha256 cannot be empty")
        if not isinstance(cache_hit, bool) or not isinstance(succeeded, bool):
            raise ValueError("cache_hit and succeeded must be boolean")
        if started_at is None:
            start = self._clock() - elapsed
        else:
            start = _nonnegative_finite(started_at, field_name="started_at")
        sample = TimingSample(
            phase=resolved,
            candidate_sha256=candidate_sha256,
            units=checked_units,
            started_at=start,
            elapsed_seconds=elapsed,
            cache_hit=cache_hit,
            succeeded=succeeded,
        )
        self._append(sample)
        return sample

    @property
    def samples(self) -> tuple[TimingSample, ...]:
        with self._lock:
            return tuple(self._samples)

    def _append(self, sample: TimingSample) -> None:
        with self._lock:
            self._samples.append(sample)

    def report(self) -> RunTimingReport:
        samples = self.samples
        if not samples:
            return RunTimingReport(0.0, 0.0, 0.0, 1.0, ())
        intervals = sorted((item.started_at, item.ended_at) for item in samples)
        active_seconds = 0.0
        interval_start, interval_end = intervals[0]
        for next_start, next_end in intervals[1:]:
            if next_start > interval_end:
                active_seconds += interval_end - interval_start
                interval_start, interval_end = next_start, next_end
            else:
                interval_end = max(interval_end, next_end)
        active_seconds += interval_end - interval_start
        task_seconds = math.fsum(item.elapsed_seconds for item in samples)
        elapsed_wall = max(item.ended_at for item in samples) - min(
            item.started_at for item in samples
        )

        phases: list[PhaseTimingStats] = []
        for phase in RunPhase:
            selected = tuple(item for item in samples if item.phase is phase)
            if not selected:
                continue
            cold = tuple(
                item for item in selected if item.succeeded and not item.cache_hit
            )
            cached = tuple(
                item for item in selected if item.succeeded and item.cache_hit
            )
            cold_rates = [item.seconds_per_unit for item in cold]
            p50 = statistics.median(cold_rates) if cold_rates else None
            p90 = _nearest_rank(cold_rates, 0.90) if cold_rates else None
            cache_saved = (
                math.fsum(
                    max(0.0, p50 * item.units - item.elapsed_seconds)
                    for item in cached
                )
                if p50 is not None
                else 0.0
            )
            phases.append(
                PhaseTimingStats(
                    phase=phase,
                    sample_count=len(selected),
                    successful_samples=sum(item.succeeded for item in selected),
                    failed_samples=sum(not item.succeeded for item in selected),
                    cold_work_units=sum(item.units for item in cold),
                    cache_hit_units=sum(item.units for item in cached),
                    elapsed_seconds=math.fsum(
                        item.elapsed_seconds for item in selected
                    ),
                    p50_seconds_per_unit=p50,
                    p90_seconds_per_unit=p90,
                    estimated_cache_seconds_saved=cache_saved,
                )
            )
        return RunTimingReport(
            elapsed_wall_seconds=elapsed_wall,
            active_wall_seconds=active_seconds,
            task_seconds=task_seconds,
            overlap_factor=(
                task_seconds / active_seconds if active_seconds > 0.0 else 1.0
            ),
            phases=tuple(phases),
        )


@dataclass(frozen=True, slots=True)
class StageCost:
    """Per-unit forecast cost with provenance."""

    p50_seconds: float
    p90_seconds: float
    source: str = "prior"
    observed_samples: int = 0

    def __post_init__(self) -> None:
        p50 = _nonnegative_finite(self.p50_seconds, field_name="p50_seconds")
        p90 = _nonnegative_finite(self.p90_seconds, field_name="p90_seconds")
        if p90 < p50:
            raise ValueError("p90_seconds must be greater than or equal to p50_seconds")
        if not self.source.strip():
            raise ValueError("source cannot be empty")
        if (
            isinstance(self.observed_samples, bool)
            or not isinstance(self.observed_samples, int)
            or self.observed_samples < 0
        ):
            raise ValueError("observed_samples must be a non-negative integer")
        object.__setattr__(self, "p50_seconds", p50)
        object.__setattr__(self, "p90_seconds", p90)


@dataclass(frozen=True, slots=True)
class GateSurvival:
    """Expected fraction that passes each gate; defaults are conservative."""

    refusal: float = 1.0
    kl: float = 1.0
    capability: float = 1.0

    def __post_init__(self) -> None:
        for field_name in ("refusal", "kl", "capability"):
            value = _nonnegative_finite(getattr(self, field_name), field_name=field_name)
            if value > 1.0:
                raise ValueError(f"{field_name} must be at most one")
            object.__setattr__(self, field_name, value)


@dataclass(frozen=True, slots=True)
class ETAEnvelope:
    serial_seconds: float
    critical_path_seconds: float
    ideal_wall_seconds: float
    list_schedule_upper_seconds: float


@dataclass(frozen=True, slots=True)
class PortfolioETA:
    """P50/P90-cost ETA interval for the currently unfinished portfolio.

    The interval bounds scheduling of the workload implied by ``GateSurvival``;
    it is not a confidence interval over uncertain gate outcomes.
    """

    candidates: int
    workers: int
    p50: ETAEnvelope
    p90: ETAEnvelope
    naive_p50_serial_seconds: float
    refusal_first_serial_speedup: float
    baseline_seconds: float | None
    baseline_speedup_lower: float | None
    baseline_speedup_upper: float | None
    prelude_units: tuple[tuple[RunPhase, int], ...]
    costs: tuple[tuple[RunPhase, StageCost], ...]


@dataclass(frozen=True, slots=True)
class EvaluationWorkItem:
    candidate_sha256: str
    stage: FunnelStage
    units: int

    def __post_init__(self) -> None:
        if not self.candidate_sha256.strip():
            raise ValueError("candidate_sha256 cannot be empty")
        if self.stage not in _FUNNEL_TO_RUN_PHASE:
            raise ValueError(f"{self.stage.value!r} is not executable work")
        _positive_units(self.units)


def _remaining_work(summary: CandidateSummary) -> EvaluationWorkItem | None:
    stage = summary.next_stage
    if stage is FunnelStage.REFUSAL:
        units = summary.refusal_total - summary.refusal_evaluated
    elif stage is FunnelStage.KL:
        units = summary.kl_total - summary.kl_evaluated
    elif stage in (FunnelStage.CAPABILITY, FunnelStage.PUBLIC_REPORT):
        units = 1
    else:
        return None
    if units < 1:
        raise ValueError(
            f"candidate {summary.candidate.sha256} has inconsistent {stage.value} progress"
        )
    return EvaluationWorkItem(summary.candidate.sha256, stage, units)


def plan_refusal_first_wave(
    summaries: Iterable[CandidateSummary],
    *,
    max_candidates: int | None = None,
    public_candidate_sha256: str | None = None,
) -> tuple[EvaluationWorkItem, ...]:
    """Return one globally refusal-first wave of independent candidate work.

    A wave contains exactly one stage.  If *any* candidate still needs refusal
    evaluation, no KL/capability/public work is returned.  Public-test work
    additionally requires ``public_candidate_sha256`` so a caller must freeze
    one winner before opening that split.
    """

    if max_candidates is not None:
        _positive_units(max_candidates, field_name="max_candidates")
    items = [item for summary in summaries if (item := _remaining_work(summary))]
    identities = [item.candidate_sha256 for item in items]
    if len(set(identities)) != len(identities):
        raise ValueError("candidate summaries must be unique")
    if not items:
        return ()
    priority = {
        FunnelStage.REFUSAL: 0,
        FunnelStage.KL: 1,
        FunnelStage.CAPABILITY: 2,
        FunnelStage.PUBLIC_REPORT: 3,
    }
    stage = min((item.stage for item in items), key=priority.__getitem__)
    selected = sorted(
        (item for item in items if item.stage is stage),
        key=lambda item: (-item.units, item.candidate_sha256),
    )
    if stage is FunnelStage.PUBLIC_REPORT:
        if public_candidate_sha256 is None:
            raise ValueError(
                "public report requires one explicitly frozen candidate"
            )
        selected = [
            item
            for item in selected
            if item.candidate_sha256 == public_candidate_sha256
        ]
        if not selected:
            raise ValueError(
                "frozen public candidate is not awaiting a public report"
            )
    if max_candidates is not None:
        selected = selected[:max_candidates]
    return tuple(selected)


_T = TypeVar("_T")


def execute_refusal_first_wave(
    work_items: Iterable[EvaluationWorkItem],
    worker: Callable[[EvaluationWorkItem], _T],
    *,
    max_workers: int = 1,
    profiler: RunTimingProfiler | None = None,
) -> tuple[_T, ...]:
    """Execute independent work items without mutating the funnel in workers.

    Results preserve input order.  Set ``max_workers`` to the number of truly
    independent runtime slots (separate GPUs/processes or I/O-bound judges).
    A single model on one GPU should normally keep ``max_workers=1``.
    """

    items = tuple(work_items)
    _positive_units(max_workers, field_name="max_workers")
    if not items:
        return ()
    stages = {item.stage for item in items}
    if len(stages) != 1:
        raise ValueError("a refusal-first wave cannot mix evaluation stages")

    def evaluate(item: EvaluationWorkItem) -> _T:
        if profiler is None:
            return worker(item)
        with profiler.measure(
            item.stage,
            candidate_sha256=item.candidate_sha256,
            units=item.units,
        ):
            return worker(item)

    bounded_workers = min(max_workers, len(items))
    if bounded_workers == 1:
        return tuple(evaluate(item) for item in items)
    with ThreadPoolExecutor(max_workers=bounded_workers) as executor:
        return tuple(executor.map(evaluate, items))


def _observed_costs(
    profiler: RunTimingProfiler | None,
    priors: Mapping[RunPhase | FunnelStage | str, StageCost | float],
) -> dict[RunPhase, StageCost]:
    costs: dict[RunPhase, StageCost] = {}
    for phase, value in priors.items():
        resolved = _run_phase(phase)
        costs[resolved] = (
            value
            if isinstance(value, StageCost)
            else StageCost(float(value), float(value), source="prior")
        )
    if profiler is None:
        return costs
    report = profiler.report()
    for stats in report.phases:
        if (
            stats.p50_seconds_per_unit is None
            or stats.p90_seconds_per_unit is None
        ):
            continue
        costs[stats.phase] = StageCost(
            stats.p50_seconds_per_unit,
            stats.p90_seconds_per_unit,
            source="observed",
            observed_samples=(
                stats.successful_samples
                - sum(
                    sample.cache_hit and sample.succeeded
                    for sample in profiler.samples
                    if sample.phase is stats.phase
                )
            ),
        )
    return costs


def _candidate_seconds(
    summary: CandidateSummary,
    costs: Mapping[RunPhase, float],
    survival: GateSurvival,
    *,
    gated: bool,
) -> float:
    stage = summary.next_stage
    refusal_cost = costs.get(RunPhase.REFUSAL, 0.0)
    kl_cost = costs.get(RunPhase.KL, 0.0)
    capability_cost = costs.get(RunPhase.CAPABILITY, 0.0)
    refusal_pass = survival.refusal if gated else 1.0
    kl_pass = survival.kl if gated else 1.0
    capability_tail = capability_cost
    kl_tail = summary.kl_total * kl_cost + kl_pass * capability_tail
    if stage is FunnelStage.REFUSAL:
        return (
            (summary.refusal_total - summary.refusal_evaluated) * refusal_cost
            + refusal_pass * kl_tail
        )
    if stage is FunnelStage.KL:
        return (
            (summary.kl_total - summary.kl_evaluated) * kl_cost
            + kl_pass * capability_tail
        )
    if stage is FunnelStage.CAPABILITY:
        return capability_tail
    if stage is FunnelStage.PUBLIC_REPORT:
        return 0.0
    return 0.0


def _public_probability(
    summary: CandidateSummary,
    survival: GateSurvival,
    *,
    gated: bool,
) -> float:
    refusal_pass = survival.refusal if gated else 1.0
    kl_pass = survival.kl if gated else 1.0
    capability_pass = survival.capability if gated else 1.0
    if summary.next_stage is FunnelStage.REFUSAL:
        return refusal_pass * kl_pass * capability_pass
    if summary.next_stage is FunnelStage.KL:
        return kl_pass * capability_pass
    if summary.next_stage is FunnelStage.CAPABILITY:
        return capability_pass
    if summary.next_stage is FunnelStage.PUBLIC_REPORT:
        return 1.0
    return 0.0


def _one_public_tail_seconds(
    summaries: Sequence[CandidateSummary],
    survival: GateSurvival,
    public_cost: float,
    *,
    gated: bool,
) -> float:
    no_winner_probability = math.prod(
        1.0 - _public_probability(summary, survival, gated=gated)
        for summary in summaries
    )
    return (1.0 - no_winner_probability) * public_cost


def _required_phases(summaries: Sequence[CandidateSummary]) -> set[RunPhase]:
    required: set[RunPhase] = set()
    for summary in summaries:
        stage = summary.next_stage
        if stage is FunnelStage.REFUSAL:
            required.update(_EVALUATION_PHASES)
        elif stage is FunnelStage.KL:
            required.update(
                (RunPhase.KL, RunPhase.CAPABILITY, RunPhase.PUBLIC_REPORT)
            )
        elif stage is FunnelStage.CAPABILITY:
            required.update((RunPhase.CAPABILITY, RunPhase.PUBLIC_REPORT))
        elif stage is FunnelStage.PUBLIC_REPORT:
            required.add(RunPhase.PUBLIC_REPORT)
    return required


def _envelope(
    chains: Sequence[float],
    *,
    workers: int,
    fixed_overhead: float,
    shared_tail: float,
) -> ETAEnvelope:
    work = math.fsum(chains) + shared_tail
    longest_chain = max(chains, default=0.0) + shared_tail
    lower = max(work / workers, longest_chain)
    # For precedence-constrained work, a work-conserving list schedule obeys
    # Cmax <= W/m + (1 - 1/m)L.  The fixed prelude cannot overlap with it.
    upper = work / workers + (1.0 - 1.0 / workers) * longest_chain
    return ETAEnvelope(
        serial_seconds=fixed_overhead + work,
        critical_path_seconds=fixed_overhead + longest_chain,
        ideal_wall_seconds=fixed_overhead + lower,
        list_schedule_upper_seconds=fixed_overhead + upper,
    )


def estimate_portfolio_eta(
    summaries: Iterable[CandidateSummary],
    *,
    profiler: RunTimingProfiler | None = None,
    priors: Mapping[RunPhase | FunnelStage | str, StageCost | float] | None = None,
    survival: GateSurvival = GateSurvival(),
    workers: int = 1,
    prelude_units: Mapping[RunPhase | str, int] | None = None,
    fixed_overhead_seconds: float = 0.0,
    fixed_overhead_p90_seconds: float | None = None,
    baseline_seconds: float | None = None,
) -> PortfolioETA:
    """Estimate remaining end-to-end time from observed per-unit phase costs.

    The default survival profile assumes every active candidate passes every
    future gate, yielding a conservative workload.  Pass historical survival
    rates only when they come from the same frozen validation protocol.
    ``prelude_units`` models serial geometry and artifact-build work explicitly;
    use it for a genuinely end-to-end forecast instead of hiding those phases
    in ``fixed_overhead_seconds``. P90 propagates per-unit P90 costs; it is not
    a calibrated whole-run quantile.
    """

    candidates = tuple(summaries)
    _positive_units(workers, field_name="workers")
    p50_overhead = _nonnegative_finite(
        fixed_overhead_seconds, field_name="fixed_overhead_seconds"
    )
    p90_overhead = (
        p50_overhead
        if fixed_overhead_p90_seconds is None
        else _nonnegative_finite(
            fixed_overhead_p90_seconds,
            field_name="fixed_overhead_p90_seconds",
        )
    )
    if p90_overhead < p50_overhead:
        raise ValueError("fixed_overhead_p90_seconds cannot be below p50 overhead")
    checked_baseline = (
        None
        if baseline_seconds is None
        else _nonnegative_finite(baseline_seconds, field_name="baseline_seconds")
    )
    checked_prelude: dict[RunPhase, int] = {}
    for phase, units in ({} if prelude_units is None else prelude_units).items():
        resolved = _run_phase(phase)
        if resolved not in (RunPhase.GEOMETRY, RunPhase.ARTIFACT_BUILD):
            raise ValueError(
                "prelude_units may contain only geometry and artifact-build"
            )
        checked_units = _nonnegative_units(
            units,
            field_name=f"prelude_units[{resolved.value!r}]",
        )
        if checked_units:
            checked_prelude[resolved] = checked_units
    costs = _observed_costs(profiler, {} if priors is None else priors)
    required = _required_phases(candidates).union(checked_prelude)
    missing = sorted(
        required.difference(costs), key=lambda phase: phase.value
    )
    if missing:
        names = ", ".join(phase.value for phase in missing)
        raise ValueError(f"missing measured cost or explicit prior for: {names}")

    p50_costs = {phase: value.p50_seconds for phase, value in costs.items()}
    p90_costs = {phase: value.p90_seconds for phase, value in costs.items()}
    p50_prelude = math.fsum(
        p50_costs[phase] * units for phase, units in checked_prelude.items()
    )
    p90_prelude = math.fsum(
        p90_costs[phase] * units for phase, units in checked_prelude.items()
    )
    p50_chains = [
        _candidate_seconds(item, p50_costs, survival, gated=True)
        for item in candidates
    ]
    p90_chains = [
        _candidate_seconds(item, p90_costs, survival, gated=True)
        for item in candidates
    ]
    p50_public_tail = _one_public_tail_seconds(
        candidates,
        survival,
        p50_costs.get(RunPhase.PUBLIC_REPORT, 0.0),
        gated=True,
    )
    p90_public_tail = _one_public_tail_seconds(
        candidates,
        survival,
        p90_costs.get(RunPhase.PUBLIC_REPORT, 0.0),
        gated=True,
    )
    naive_public_tail = _one_public_tail_seconds(
        candidates,
        survival,
        p50_costs.get(RunPhase.PUBLIC_REPORT, 0.0),
        gated=False,
    )
    naive_p50 = p50_overhead + p50_prelude + naive_public_tail + math.fsum(
        _candidate_seconds(item, p50_costs, survival, gated=False)
        for item in candidates
    )
    p50 = _envelope(
        p50_chains,
        workers=workers,
        fixed_overhead=p50_overhead + p50_prelude,
        shared_tail=p50_public_tail,
    )
    p90 = _envelope(
        p90_chains,
        workers=workers,
        fixed_overhead=p90_overhead + p90_prelude,
        shared_tail=p90_public_tail,
    )
    funnel_speedup = (
        naive_p50 / p50.serial_seconds if p50.serial_seconds > 0.0 else 1.0
    )
    baseline_speedup_lower = (
        checked_baseline / p90.list_schedule_upper_seconds
        if checked_baseline is not None and p90.list_schedule_upper_seconds > 0.0
        else None
    )
    baseline_speedup_upper = (
        checked_baseline / p50.ideal_wall_seconds
        if checked_baseline is not None and p50.ideal_wall_seconds > 0.0
        else None
    )
    return PortfolioETA(
        candidates=len(candidates),
        workers=workers,
        p50=p50,
        p90=p90,
        naive_p50_serial_seconds=naive_p50,
        refusal_first_serial_speedup=funnel_speedup,
        baseline_seconds=checked_baseline,
        baseline_speedup_lower=baseline_speedup_lower,
        baseline_speedup_upper=baseline_speedup_upper,
        prelude_units=tuple(
            sorted(checked_prelude.items(), key=lambda item: item[0].value)
        ),
        costs=tuple(sorted(costs.items(), key=lambda item: item[0].value)),
    )


__all__ = [
    "ETAEnvelope",
    "EvaluationWorkItem",
    "GateSurvival",
    "PhaseTimingStats",
    "PortfolioETA",
    "RunPhase",
    "RunTimingProfiler",
    "RunTimingReport",
    "StageCost",
    "TimingSample",
    "estimate_portfolio_eta",
    "execute_refusal_first_wave",
    "plan_refusal_first_wave",
]
