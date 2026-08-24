"""Fail-closed benign-intent oracle aggregation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OracleVerdict:
    oracle_id: str
    benign: bool | None
    confidence: float
    rationale: str = ""


def benign_consensus(
    verdicts: list[OracleVerdict] | tuple[OracleVerdict, ...],
    *,
    confidence_minimum: float = 0.8,
) -> bool:
    """Accept only unanimous, confident benign verdicts; ambiguity fails closed."""

    if len(verdicts) < 2:
        return False
    return all(
        verdict.benign is True and verdict.confidence >= confidence_minimum
        for verdict in verdicts
    )
