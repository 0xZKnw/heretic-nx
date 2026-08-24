"""Benign task-success metrics that resist refusal-marker gaming."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskOutcome:
    task: str
    success: bool
    refusal: bool
    deflection: bool
    task_score: float


@dataclass(frozen=True)
class BenignMetrics:
    count: int
    benign_success_rate: float
    benign_refusal_rate: float
    deflection_rate: float
    mean_task_score: float
    by_task: dict[str, "BenignMetrics"]


def aggregate_benign_metrics(
    outcomes: list[TaskOutcome] | tuple[TaskOutcome, ...],
    *,
    include_by_task: bool = True,
) -> BenignMetrics:
    if not outcomes:
        raise ValueError("at least one task outcome is required")
    count = len(outcomes)
    groups: dict[str, list[TaskOutcome]] = {}
    if include_by_task:
        for outcome in outcomes:
            groups.setdefault(outcome.task, []).append(outcome)
    return BenignMetrics(
        count=count,
        benign_success_rate=sum(item.success for item in outcomes) / count,
        benign_refusal_rate=sum(item.refusal for item in outcomes) / count,
        deflection_rate=sum(item.deflection for item in outcomes) / count,
        mean_task_score=sum(item.task_score for item in outcomes) / count,
        by_task={
            task: aggregate_benign_metrics(rows, include_by_task=False)
            for task, rows in groups.items()
        },
    )
