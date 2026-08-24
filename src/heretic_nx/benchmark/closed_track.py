"""Preregistered, paired closed-track comparison against pinned engines."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator

from heretic_nx.eval.capability import PairedInterval, paired_bootstrap_interval
from heretic_nx.hashing import sha256_json


SHA256_PATTERN = r"^[0-9a-f]{64}$"


class OptimizationBudget(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    maximum_trials: int = Field(ge=1)
    maximum_forward_tokens: int = Field(ge=1)
    maximum_backward_tokens: int = Field(ge=0)
    maximum_gpu_seconds: float = Field(gt=0)
    maximum_peak_vram_bytes: int = Field(ge=1)


class ArmRegistration(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    role: Literal["candidate", "competitor"]
    engine_id: str
    engine_revision: str
    engine_config_sha256: str = Field(pattern=SHA256_PATTERN)
    base_model_sha256: str = Field(pattern=SHA256_PATTERN)
    training_data_sha256: str = Field(pattern=SHA256_PATTERN)
    split_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    budget: OptimizationBudget
    seeds: tuple[int, ...] = Field(min_length=1)


class ClosedTrackRegistration(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    track_id: str
    benchmark_data_sha256: str = Field(pattern=SHA256_PATTERN)
    judge_rubric_sha256: str = Field(pattern=SHA256_PATTERN)
    capability_margin: float = Field(ge=0)
    risk_margin: float = Field(ge=0)
    target_superiority_margin: float = Field(ge=0)
    alpha: float = Field(gt=0, lt=1)
    resamples: int = Field(ge=100)
    arms: tuple[ArmRegistration, ...] = Field(min_length=2)

    @model_validator(mode="after")
    def validate_matched_track(self) -> "ClosedTrackRegistration":
        ids = {arm.id for arm in self.arms}
        if len(ids) != len(self.arms):
            raise ValueError("closed-track arm ids must be unique")
        candidates = [arm for arm in self.arms if arm.role == "candidate"]
        competitors = [arm for arm in self.arms if arm.role == "competitor"]
        if len(candidates) != 1 or not competitors:
            raise ValueError("closed track requires exactly one candidate and competitors")
        candidate = candidates[0]
        matched_fields = (
            "base_model_sha256",
            "training_data_sha256",
            "split_manifest_sha256",
            "budget",
            "seeds",
        )
        for competitor in competitors:
            mismatched = [
                field
                for field in matched_fields
                if getattr(candidate, field) != getattr(competitor, field)
            ]
            if mismatched:
                raise ValueError(
                    f"arm {competitor.id} is not budget/data matched: {mismatched}"
                )
        return self

    @property
    def content_id(self) -> str:
        return sha256_json(self.model_dump())

    def write(self, path: str) -> None:
        from pathlib import Path
        from heretic_nx.hashing import canonical_json

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(canonical_json(self.model_dump()) + b"\n")


class ArmObservations(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    arm_id: str
    output_model_sha256: str = Field(pattern=SHA256_PATTERN)
    response_artifact_sha256: str = Field(pattern=SHA256_PATTERN)
    target_scores: dict[str, float]
    capability_scores: dict[str, float]
    risk_scores: dict[str, float]
    trials_used: int = Field(ge=0)
    forward_tokens_used: int = Field(ge=0)
    backward_tokens_used: int = Field(ge=0)
    gpu_seconds_used: float = Field(ge=0)
    peak_vram_bytes: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_scores(self) -> "ArmObservations":
        for metric, values in (
            ("target", self.target_scores),
            ("capability", self.capability_scores),
            ("risk", self.risk_scores),
        ):
            if len(values) < 2:
                raise ValueError(f"{metric} observations require at least two items")
            if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values.values()):
                raise ValueError(f"{metric} observations must be finite scores in [0, 1]")
        return self


@dataclass(frozen=True)
class CompetitorComparison:
    competitor_id: str
    target: PairedInterval
    capability: PairedInterval
    risk_safety: PairedInterval
    target_superiority_passed: bool
    capability_noninferiority_passed: bool
    risk_noninferiority_passed: bool
    passed: bool


@dataclass(frozen=True)
class ClosedTrackResult:
    registration_sha256: str
    benchmark_winner: bool
    comparisons: tuple[CompetitorComparison, ...]
    blocking_reasons: tuple[str, ...]


def _aligned_values(
    candidate: Mapping[str, float],
    competitor: Mapping[str, float],
    metric: str,
) -> tuple[list[float], list[float]]:
    if set(candidate) != set(competitor) or len(candidate) < 2:
        raise ValueError(f"{metric} observations must have identical item ids")
    ids = sorted(candidate)
    return [competitor[item] for item in ids], [candidate[item] for item in ids]


def _budget_blockers(arm: ArmRegistration, observed: ArmObservations) -> list[str]:
    budget = arm.budget
    used = {
        "trials": (observed.trials_used, budget.maximum_trials),
        "forward-tokens": (observed.forward_tokens_used, budget.maximum_forward_tokens),
        "backward-tokens": (observed.backward_tokens_used, budget.maximum_backward_tokens),
        "gpu-seconds": (observed.gpu_seconds_used, budget.maximum_gpu_seconds),
        "peak-vram": (observed.peak_vram_bytes, budget.maximum_peak_vram_bytes),
    }
    return [name for name, (value, maximum) in used.items() if value > maximum]


def evaluate_closed_track(
    registration: ClosedTrackRegistration,
    observations: Mapping[str, ArmObservations],
    *,
    seed: int = 0,
) -> ClosedTrackResult:
    registered = {arm.id: arm for arm in registration.arms}
    if set(observations) != set(registered):
        raise ValueError("observations must match every registered arm exactly")
    for arm_id, observed in observations.items():
        if arm_id != observed.arm_id:
            raise ValueError(f"observation key {arm_id} does not match its arm id")
    candidate_registration = next(arm for arm in registration.arms if arm.role == "candidate")
    candidate = observations[candidate_registration.id]
    blockers = [
        f"budget:{candidate_registration.id}:{name}"
        for name in _budget_blockers(candidate_registration, candidate)
    ]
    competitors = [arm for arm in registration.arms if arm.role == "competitor"]
    comparison_alpha = registration.alpha / (3 * len(competitors))
    comparisons = []
    for index, competitor_registration in enumerate(competitors):
        competitor = observations[competitor_registration.id]
        blockers.extend(
            f"budget:{competitor_registration.id}:{name}"
            for name in _budget_blockers(competitor_registration, competitor)
        )
        target_base, target_candidate = _aligned_values(
            candidate.target_scores, competitor.target_scores, "target"
        )
        capability_base, capability_candidate = _aligned_values(
            candidate.capability_scores, competitor.capability_scores, "capability"
        )
        risk_base, risk_candidate = _aligned_values(
            {key: 1.0 - value for key, value in candidate.risk_scores.items()},
            {key: 1.0 - value for key, value in competitor.risk_scores.items()},
            "risk",
        )
        target = paired_bootstrap_interval(
            target_base,
            target_candidate,
            margin=registration.target_superiority_margin,
            alpha=comparison_alpha,
            resamples=registration.resamples,
            seed=seed + index * 3,
        )
        capability = paired_bootstrap_interval(
            capability_base,
            capability_candidate,
            margin=registration.capability_margin,
            alpha=comparison_alpha,
            resamples=registration.resamples,
            seed=seed + index * 3 + 1,
        )
        risk = paired_bootstrap_interval(
            risk_base,
            risk_candidate,
            margin=registration.risk_margin,
            alpha=comparison_alpha,
            resamples=registration.resamples,
            seed=seed + index * 3 + 2,
        )
        target_passed = target.lower > registration.target_superiority_margin
        capability_passed = capability.noninferiority_passed
        risk_passed = risk.noninferiority_passed
        passed = target_passed and capability_passed and risk_passed
        if not passed:
            blockers.append(f"comparison:{competitor_registration.id}")
        comparisons.append(
            CompetitorComparison(
                competitor_registration.id,
                target,
                capability,
                risk,
                target_passed,
                capability_passed,
                risk_passed,
                passed,
            )
        )
    return ClosedTrackResult(
        registration_sha256=registration.content_id,
        benchmark_winner=not blockers,
        comparisons=tuple(comparisons),
        blocking_reasons=tuple(blockers),
    )
