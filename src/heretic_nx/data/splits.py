"""Deterministic anti-leak split assignment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from heretic_nx.hashing import sha256_json


SplitName = Literal[
    "train-geometry",
    "search-public",
    "secret-b",
    "secret-h",
    "cross-template",
    "cross-quant",
]


@dataclass(frozen=True)
class SplitAssignment:
    item_id: str
    split: SplitName
    assignment_hash: str


def assign_split(
    item_id: str,
    *,
    seed: int,
    risk_canary: bool = False,
    cross_template: bool = False,
    cross_quant: bool = False,
) -> SplitAssignment:
    digest = sha256_json({"item_id": item_id, "seed": seed})
    if risk_canary:
        split: SplitName = "secret-h"
    elif cross_template:
        split = "cross-template"
    elif cross_quant:
        split = "cross-quant"
    else:
        bucket = int(digest[:8], 16) % 100
        split = "train-geometry" if bucket < 60 else "search-public" if bucket < 80 else "secret-b"
    return SplitAssignment(item_id, split, digest)


def validate_no_leakage(assignments: list[SplitAssignment] | tuple[SplitAssignment, ...]) -> None:
    seen: dict[str, SplitName] = {}
    for assignment in assignments:
        previous = seen.get(assignment.item_id)
        if previous is not None and previous != assignment.split:
            raise ValueError(
                f"item {assignment.item_id} appears in both {previous} and {assignment.split}"
            )
        seen[assignment.item_id] = assignment.split
