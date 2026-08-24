"""Gradient attribution pre-scan and exact central-difference confirmation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class AttributionScore:
    site_id: str
    raw_gradient: float
    adjusted_score: float


@dataclass(frozen=True)
class ReliabilityDiagnostic:
    first_order: float
    central_difference: float
    relative_disagreement: float
    flagged: bool


def gradient_attribution_scores(
    loss: Tensor,
    site_parameters: Mapping[str, Tensor],
    *,
    costs: Mapping[str, float] | None = None,
    conflict_penalties: Mapping[str, float] | None = None,
) -> tuple[AttributionScore, ...]:
    if loss.ndim != 0:
        raise ValueError("loss must be scalar")
    identifiers = tuple(site_parameters)
    gradients = torch.autograd.grad(
        loss,
        [site_parameters[site_id] for site_id in identifiers],
        retain_graph=True,
        allow_unused=True,
    )
    rows = []
    for site_id, gradient in zip(identifiers, gradients):
        raw = 0.0 if gradient is None else float(gradient.detach().item())
        cost = max((costs or {}).get(site_id, 1.0), 1e-12)
        penalty = max((conflict_penalties or {}).get(site_id, 0.0), 0.0)
        rows.append(AttributionScore(site_id, raw, abs(raw) / (cost * (1.0 + penalty))))
    return tuple(sorted(rows, key=lambda row: (-row.adjusted_score, row.site_id)))


def central_difference(minus: float, plus: float, epsilon: float) -> float:
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    return (plus - minus) / (2.0 * epsilon)


def attribution_reliability(
    first_order: float,
    minus: float,
    plus: float,
    epsilon: float,
    *,
    disagreement_threshold: float = 0.35,
    floor: float = 1e-8,
) -> ReliabilityDiagnostic:
    exact = central_difference(minus, plus, epsilon)
    disagreement = abs(first_order - exact) / max(abs(first_order), abs(exact), floor)
    return ReliabilityDiagnostic(
        first_order,
        exact,
        disagreement,
        disagreement > disagreement_threshold,
    )


def select_top_k(scores: Sequence[AttributionScore], k: int) -> tuple[AttributionScore, ...]:
    if k < 1:
        raise ValueError("k must be positive")
    return tuple(scores[: min(k, len(scores))])
