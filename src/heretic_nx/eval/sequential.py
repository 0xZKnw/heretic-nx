"""Anytime-valid confidence sequences and sequential promotion rules."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal


def clopper_pearson_zero_upper(sample_count: int, alpha: float = 0.05) -> float:
    if sample_count < 1 or not 0 < alpha < 1:
        raise ValueError("invalid sample_count or alpha")
    return 1.0 - alpha ** (1.0 / sample_count)


@dataclass(frozen=True)
class ConfidenceInterval:
    lower: float
    upper: float
    count: int
    mean: float


@dataclass
class AnytimeBernoulliCS:
    """Time-uniform Hoeffding CS via a summable alpha-spending sequence."""

    alpha: float = 0.05
    successes: int = 0
    count: int = 0

    def update(self, outcome: bool) -> ConfidenceInterval:
        self.successes += int(outcome)
        self.count += 1
        return self.interval()

    def interval(self) -> ConfidenceInterval:
        if self.count == 0:
            return ConfidenceInterval(0.0, 1.0, 0, 0.0)
        mean = self.successes / self.count
        alpha_n = self.alpha * 6.0 / (math.pi**2 * self.count**2)
        radius = math.sqrt(math.log(2.0 / alpha_n) / (2.0 * self.count))
        return ConfidenceInterval(
            max(0.0, mean - radius),
            min(1.0, mean + radius),
            self.count,
            mean,
        )


@dataclass(frozen=True)
class SequentialDecision:
    action: Literal["continue", "promote", "prune", "constraint-violation"]
    reason: str


def sequential_decision(
    candidate: ConfidenceInterval,
    *,
    incumbent: ConfidenceInterval | None = None,
    higher_is_better: bool = True,
    minimum_margin: float = 0.0,
    bad_rate_interval: ConfidenceInterval | None = None,
    bad_rate_maximum: float | None = None,
) -> SequentialDecision:
    if bad_rate_interval is not None and bad_rate_maximum is not None:
        if bad_rate_interval.lower > bad_rate_maximum:
            return SequentialDecision("constraint-violation", "bad-rate lower bound exceeds limit")
    if incumbent is None:
        return SequentialDecision("continue", "no incumbent bound")
    if higher_is_better:
        if candidate.upper < incumbent.lower + minimum_margin:
            return SequentialDecision("prune", "candidate is anytime-dominated")
        if candidate.lower > incumbent.upper + minimum_margin:
            return SequentialDecision("promote", "candidate is anytime-superior")
    else:
        if candidate.lower > incumbent.upper - minimum_margin:
            return SequentialDecision("prune", "candidate is anytime-dominated")
        if candidate.upper < incumbent.lower - minimum_margin:
            return SequentialDecision("promote", "candidate is anytime-superior")
    return SequentialDecision("continue", "confidence sequences overlap")
