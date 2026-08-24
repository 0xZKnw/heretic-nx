"""Deterministic group-aware anti-leak split assignment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from heretic_nx.hashing import sha256_json


SplitName = Literal[
    "train-geometry",
    "validation-search",
    "public-test",
    "secret-b",
    "secret-h",
    "cross-template",
    "cross-quant",
]

PhaseName = Literal["geometry", "selection", "public-report", "secret-audit"]


@dataclass(frozen=True)
class SplitAssignment:
    item_id: str
    split: SplitName
    assignment_hash: str
    group_id: str | None = None


def assign_split(
    item_id: str,
    *,
    seed: int,
    group_id: str | None = None,
    risk_canary: bool = False,
    cross_template: bool = False,
    cross_quant: bool = False,
) -> SplitAssignment:
    if sum((risk_canary, cross_template, cross_quant)) > 1:
        raise ValueError("an item cannot belong to multiple reserved split families")
    semantic_group = group_id or item_id
    digest = sha256_json({"group_id": semantic_group, "seed": seed})
    if risk_canary:
        split: SplitName = "secret-h"
    elif cross_template:
        split = "cross-template"
    elif cross_quant:
        split = "cross-quant"
    else:
        bucket = int(digest[:8], 16) % 100
        if bucket < 50:
            split = "train-geometry"
        elif bucket < 70:
            split = "validation-search"
        elif bucket < 85:
            split = "public-test"
        else:
            split = "secret-b"
    return SplitAssignment(item_id, split, digest, semantic_group)


def validate_no_leakage(assignments: list[SplitAssignment] | tuple[SplitAssignment, ...]) -> None:
    seen: dict[str, SplitName] = {}
    seen_groups: dict[str, SplitName] = {}
    for assignment in assignments:
        previous = seen.get(assignment.item_id)
        if previous is not None and previous != assignment.split:
            raise ValueError(
                f"item {assignment.item_id} appears in both {previous} and {assignment.split}"
            )
        seen[assignment.item_id] = assignment.split
        group_id = assignment.group_id or assignment.item_id
        previous_group = seen_groups.get(group_id)
        if previous_group is not None and previous_group != assignment.split:
            raise ValueError(
                f"semantic group {group_id} appears in both {previous_group} "
                f"and {assignment.split}"
            )
        seen_groups[group_id] = assignment.split


def assert_phase_allowed(assignments: list[SplitAssignment] | tuple[SplitAssignment, ...], phase: PhaseName) -> None:
    """Fail closed when a pipeline phase touches an unauthorized partition."""

    allowed: dict[PhaseName, frozenset[SplitName]] = {
        "geometry": frozenset({"train-geometry"}),
        "selection": frozenset({"validation-search"}),
        "public-report": frozenset({"public-test", "cross-template", "cross-quant"}),
        "secret-audit": frozenset({"secret-b", "secret-h"}),
    }
    invalid = sorted({assignment.split for assignment in assignments} - allowed[phase])
    if invalid:
        raise ValueError(f"phase {phase} cannot access partitions: {invalid}")
