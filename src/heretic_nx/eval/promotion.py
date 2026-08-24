"""Mechanical PRIME claim derivation from content-addressed gate evidence."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Literal, Mapping


GateStatus = Literal["pass", "fail", "not-run"]
PrimeClaim = Literal[
    "ineligible",
    "PRIME-candidate",
    "PRIME-validated",
    "PRIME-reproduced",
    "benchmark-winner",
]

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TIERS: tuple[tuple[PrimeClaim, tuple[str, ...]], ...] = (
    ("PRIME-candidate", ("geometry", "causal", "behavior")),
    ("PRIME-validated", ("capability", "provenance")),
    ("PRIME-reproduced", ("reproduction",)),
    ("benchmark-winner", ("benchmark",)),
)


@dataclass(frozen=True)
class GateEvidence:
    gate: str
    status: GateStatus
    artifact_sha256: str | None
    reason: str

    def __post_init__(self) -> None:
        if self.status == "pass" and (
            self.artifact_sha256 is None or not _SHA256.fullmatch(self.artifact_sha256)
        ):
            raise ValueError("passing gate evidence requires a lowercase SHA-256 artifact")
        if not self.reason.strip():
            raise ValueError("gate evidence requires a reason")


@dataclass(frozen=True)
class PrimePromotionResult:
    claim: PrimeClaim
    passed_gates: tuple[str, ...]
    blocking_gates: tuple[str, ...]
    external_accreditation: bool = False


def derive_prime_claim(evidence: Mapping[str, GateEvidence]) -> PrimePromotionResult:
    """Return the highest contiguous claim tier supported by immutable evidence.

    PRIME is a project validation protocol, not an external accreditation. A
    higher tier cannot skip a missing or failed lower tier.
    """

    for key, item in evidence.items():
        if key != item.gate:
            raise ValueError(f"evidence key {key} does not match gate {item.gate}")

    claim: PrimeClaim = "ineligible"
    passed: list[str] = []
    blocking: list[str] = []
    for tier, required in _TIERS:
        tier_passed = True
        for gate in required:
            item = evidence.get(gate)
            if item is None or item.status != "pass":
                tier_passed = False
                blocking.append(gate)
            else:
                passed.append(gate)
        if not tier_passed:
            break
        claim = tier
    return PrimePromotionResult(claim, tuple(passed), tuple(blocking))
