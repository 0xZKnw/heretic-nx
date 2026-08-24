"""Safety-preserving delta debugging for benign false-refusal triggers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from difflib import SequenceMatcher


@dataclass(frozen=True)
class BoundaryMiningResult:
    original: str
    minimized: str
    removed_fragments: tuple[str, ...]
    predicate_calls: int
    safety_calls: int


def _removed_fragments(original: list[str], minimized: list[str]) -> tuple[str, ...]:
    matcher = SequenceMatcher(a=original, b=minimized)
    removed = []
    for tag, i1, i2, _j1, _j2 in matcher.get_opcodes():
        if tag in {"delete", "replace"} and i1 != i2:
            removed.append(" ".join(original[i1:i2]))
    return tuple(removed)


def delta_debug_benign_refusal(
    prompt: str,
    *,
    refusal_predicate: Callable[[str], bool],
    benign_oracle: Callable[[str], bool],
    minimum_tokens: int = 2,
    maximum_calls: int = 128,
) -> BoundaryMiningResult:
    """Minimize a benign refused prompt while revalidating benign intent every step."""

    tokens = prompt.split()
    if len(tokens) < minimum_tokens:
        raise ValueError("prompt is shorter than minimum_tokens")
    predicate_calls = 0
    safety_calls = 1
    if not benign_oracle(prompt):
        raise ValueError("original prompt is not confidently benign")
    predicate_calls += 1
    if not refusal_predicate(prompt):
        raise ValueError("original prompt does not trigger refusal")

    current = tokens
    granularity = 2
    while len(current) > minimum_tokens and predicate_calls + safety_calls < maximum_calls:
        chunk_size = max(1, (len(current) + granularity - 1) // granularity)
        reduced = False
        for start in range(0, len(current), chunk_size):
            candidate_tokens = current[:start] + current[start + chunk_size :]
            if len(candidate_tokens) < minimum_tokens:
                continue
            candidate = " ".join(candidate_tokens)
            safety_calls += 1
            if not benign_oracle(candidate):
                continue
            predicate_calls += 1
            if refusal_predicate(candidate):
                current = candidate_tokens
                granularity = max(2, granularity - 1)
                reduced = True
                break
            if predicate_calls + safety_calls >= maximum_calls:
                break
        if not reduced:
            if granularity >= len(current):
                break
            granularity = min(len(current), granularity * 2)

    return BoundaryMiningResult(
        original=prompt,
        minimized=" ".join(current),
        removed_fragments=_removed_fragments(tokens, current),
        predicate_calls=predicate_calls,
        safety_calls=safety_calls,
    )
