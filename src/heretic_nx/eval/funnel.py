"""Content-addressed, refusal-first evaluation funnel.

The funnel is deliberately an orchestration primitive rather than an evaluator.
It never calls a model or a judge.  Callers submit immutable, content-addressed
observations produced by deterministic rules, task oracles, or human review.

The enforced order is::

    complete validation-search refusal evaluation (and pass its cap)
        -> complete KL evaluation (and pass its cap)
        -> capability gate
        -> public-test report

This makes it impossible for a normal caller to spend KL work on an unqualified
candidate or to use public-test rows during candidate selection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
import math
import re
from typing import Iterable

from heretic_nx.hashing import sha256_json


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class FunnelError(RuntimeError):
    """Base class for funnel integrity and ordering failures."""


class IdentityError(FunnelError, ValueError):
    """Raised when content identities or split contracts are inconsistent."""


class StageOrderError(FunnelError):
    """Raised when evidence is submitted before its prerequisite gate."""


class CacheConflictError(FunnelError):
    """Raised when an immutable cache slot receives different evidence."""


class SplitRole(StrEnum):
    TRAIN_GEOMETRY = "train-geometry"
    VALIDATION_SEARCH = "validation-search"
    PUBLIC_TEST = "public-test"


class SemanticLabel(StrEnum):
    COMPLIANCE = "compliance"
    REFUSAL = "refusal"
    PARTIAL_REFUSAL = "partial-refusal"
    DEFLECTION = "deflection"


class VerdictSource(StrEnum):
    """Permitted semantic evidence sources; network judges are not a dependency."""

    DETERMINISTIC = "deterministic"
    TASK_ORACLE = "task-oracle"
    HUMAN = "human"


class FunnelStage(StrEnum):
    REFUSAL = "refusal"
    KL = "kl"
    CAPABILITY = "capability"
    PUBLIC_REPORT = "public-report"
    COMPLETE = "complete"
    REJECTED = "rejected"


def _checked_sha256(value: str, *, field_name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise IdentityError(f"{field_name} must be a lowercase SHA-256")
    return value


@dataclass(frozen=True, slots=True)
class SplitItem:
    """Identity-bearing row in a frozen split manifest."""

    item_id: str
    group_id: str
    prompt_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.item_id, str) or not self.item_id.strip():
            raise IdentityError("split item_id cannot be empty")
        if not isinstance(self.group_id, str) or not self.group_id.strip():
            raise IdentityError("split group_id cannot be empty")
        _checked_sha256(self.prompt_sha256, field_name="prompt_sha256")

    def canonical_payload(self) -> dict[str, str]:
        return {
            "group_id": self.group_id,
            "item_id": self.item_id,
            "prompt_sha256": self.prompt_sha256,
        }


@dataclass(frozen=True, slots=True)
class SplitManifest:
    """Frozen split whose hash covers role, row order, groups, and prompts."""

    role: SplitRole
    items: tuple[SplitItem, ...]

    def __post_init__(self) -> None:
        try:
            role = SplitRole(self.role)
        except ValueError as error:
            raise IdentityError(f"unsupported split role: {self.role!r}") from error
        items = tuple(self.items)
        if not items:
            raise IdentityError("split manifest cannot be empty")
        if not all(isinstance(item, SplitItem) for item in items):
            raise IdentityError("split manifest items must be SplitItem instances")
        item_ids = [item.item_id for item in items]
        if len(set(item_ids)) != len(item_ids):
            raise IdentityError("split item IDs must be unique")
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "items", items)

    @property
    def sha256(self) -> str:
        return sha256_json(
            {
                "items": [item.canonical_payload() for item in self.items],
                "role": self.role.value,
                "schema_version": "heretic-nx-split-manifest-v1",
            }
        )

    @property
    def item_ids(self) -> tuple[str, ...]:
        return tuple(item.item_id for item in self.items)


@dataclass(frozen=True, slots=True)
class FunnelProtocol:
    """All evaluator semantics that must invalidate cached observations."""

    runtime_sha256: str
    tokenizer_sha256: str
    chat_template_sha256: str
    generation_config_sha256: str
    lexical_rule_sha256: str
    semantic_rule_sha256: str
    kl_spec_sha256: str
    capability_spec_sha256: str
    refusal_cap: int
    kl_cap: float
    refusal_labels: tuple[SemanticLabel, ...] = (
        SemanticLabel.REFUSAL,
        SemanticLabel.PARTIAL_REFUSAL,
        SemanticLabel.DEFLECTION,
    )

    def __post_init__(self) -> None:
        for field_name in (
            "runtime_sha256",
            "tokenizer_sha256",
            "chat_template_sha256",
            "generation_config_sha256",
            "lexical_rule_sha256",
            "semantic_rule_sha256",
            "kl_spec_sha256",
            "capability_spec_sha256",
        ):
            _checked_sha256(getattr(self, field_name), field_name=field_name)
        if (
            isinstance(self.refusal_cap, bool)
            or not isinstance(self.refusal_cap, int)
            or self.refusal_cap < 0
        ):
            raise ValueError("refusal_cap must be a non-negative integer")
        if (
            isinstance(self.kl_cap, bool)
            or not isinstance(self.kl_cap, (int, float))
            or not math.isfinite(float(self.kl_cap))
            or float(self.kl_cap) < 0.0
        ):
            raise ValueError("kl_cap must be a finite non-negative number")
        try:
            labels = tuple(
                sorted(
                    {SemanticLabel(label) for label in self.refusal_labels},
                    key=lambda label: label.value,
                )
            )
        except ValueError as error:
            raise ValueError("refusal_labels contains an unsupported label") from error
        if not labels or SemanticLabel.COMPLIANCE in labels:
            raise ValueError("refusal_labels must contain failures, not compliance")
        object.__setattr__(self, "kl_cap", float(self.kl_cap))
        object.__setattr__(self, "refusal_labels", labels)

    @property
    def sha256(self) -> str:
        return sha256_json(
            {
                "capability_spec_sha256": self.capability_spec_sha256,
                "chat_template_sha256": self.chat_template_sha256,
                "generation_config_sha256": self.generation_config_sha256,
                "kl_cap": self.kl_cap,
                "kl_spec_sha256": self.kl_spec_sha256,
                "lexical_rule_sha256": self.lexical_rule_sha256,
                "refusal_cap": self.refusal_cap,
                "refusal_labels": [label.value for label in self.refusal_labels],
                "runtime_sha256": self.runtime_sha256,
                "schema_version": "heretic-nx-funnel-protocol-v1",
                "semantic_rule_sha256": self.semantic_rule_sha256,
                "tokenizer_sha256": self.tokenizer_sha256,
            }
        )


@dataclass(frozen=True, slots=True)
class CandidateKey:
    """Content address for one artifact under one three-way split contract."""

    artifact_sha256: str
    protocol_sha256: str
    geometry_split_sha256: str
    search_split_sha256: str
    public_test_split_sha256: str

    def __post_init__(self) -> None:
        for field_name in (
            "artifact_sha256",
            "protocol_sha256",
            "geometry_split_sha256",
            "search_split_sha256",
            "public_test_split_sha256",
        ):
            _checked_sha256(getattr(self, field_name), field_name=field_name)
        split_hashes = {
            self.geometry_split_sha256,
            self.search_split_sha256,
            self.public_test_split_sha256,
        }
        if len(split_hashes) != 3:
            raise IdentityError("geometry, search, and public-test hashes must differ")

    @property
    def sha256(self) -> str:
        return sha256_json(
            {
                "artifact_sha256": self.artifact_sha256,
                "geometry_split_sha256": self.geometry_split_sha256,
                "protocol_sha256": self.protocol_sha256,
                "public_test_split_sha256": self.public_test_split_sha256,
                "schema_version": "heretic-nx-funnel-candidate-v1",
                "search_split_sha256": self.search_split_sha256,
            }
        )


@dataclass(frozen=True, slots=True)
class RefusalObservation:
    """Lexical proxy and resolved semantic verdict for one generated response."""

    item_id: str
    response_sha256: str
    lexical_proxy_hit: bool
    semantic_label: SemanticLabel
    semantic_source: VerdictSource
    semantic_evidence_sha256: str

    def __post_init__(self) -> None:
        if not self.item_id.strip():
            raise ValueError("refusal observation item_id cannot be empty")
        if not isinstance(self.lexical_proxy_hit, bool):
            raise ValueError("lexical_proxy_hit must be boolean")
        _checked_sha256(self.response_sha256, field_name="response_sha256")
        _checked_sha256(
            self.semantic_evidence_sha256,
            field_name="semantic_evidence_sha256",
        )
        try:
            label = SemanticLabel(self.semantic_label)
            source = VerdictSource(self.semantic_source)
        except ValueError as error:
            raise ValueError("unsupported semantic label or verdict source") from error
        object.__setattr__(self, "semantic_label", label)
        object.__setattr__(self, "semantic_source", source)


@dataclass(frozen=True, slots=True)
class KLObservation:
    item_id: str
    value: float
    evidence_sha256: str

    def __post_init__(self) -> None:
        if not self.item_id.strip():
            raise ValueError("KL observation item_id cannot be empty")
        if (
            isinstance(self.value, bool)
            or not isinstance(self.value, (int, float))
            or not math.isfinite(float(self.value))
            or float(self.value) < 0.0
        ):
            raise ValueError("KL must be a finite non-negative number")
        _checked_sha256(self.evidence_sha256, field_name="evidence_sha256")
        object.__setattr__(self, "value", float(self.value))


@dataclass(frozen=True, slots=True)
class CapabilityObservation:
    passed: bool
    evidence_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.passed, bool):
            raise ValueError("capability passed must be boolean")
        _checked_sha256(self.evidence_sha256, field_name="evidence_sha256")


@dataclass(frozen=True, slots=True)
class PublicReportObservation:
    evidence_sha256: str

    def __post_init__(self) -> None:
        _checked_sha256(self.evidence_sha256, field_name="evidence_sha256")


@dataclass(frozen=True, slots=True)
class FunnelResult:
    candidate_sha256: str
    cache_hit: bool
    next_stage: FunnelStage


@dataclass(frozen=True, slots=True)
class RegistrationResult(FunnelResult):
    candidate: CandidateKey


@dataclass(frozen=True, slots=True)
class CacheStats:
    hits: int
    misses: int

    @property
    def attempts(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return 0.0 if self.attempts == 0 else self.hits / self.attempts


@dataclass(frozen=True, slots=True)
class CandidateSummary:
    candidate: CandidateKey
    next_stage: FunnelStage
    refusal_evaluated: int
    refusal_total: int
    lexical_proxy_hits: int
    semantic_refusals: int
    kl_evaluated: int
    kl_total: int
    kl_sum: float
    kl_lower_bound: float
    mean_kl: float | None
    capability_passed: bool | None
    public_reported: bool


@dataclass(frozen=True, slots=True)
class FrontierPoint:
    candidate_sha256: str
    artifact_sha256: str
    semantic_refusals: int
    mean_kl: float


@dataclass(slots=True)
class _CandidateState:
    candidate: CandidateKey
    protocol: FunnelProtocol
    search_item_ids: tuple[str, ...]
    refusals: dict[str, RefusalObservation] = field(default_factory=dict)
    kl_rows: dict[str, KLObservation] = field(default_factory=dict)
    capability: CapabilityObservation | None = None
    public_report: PublicReportObservation | None = None


def _require_no_split_leakage(manifests: Iterable[SplitManifest]) -> None:
    item_owner: dict[str, SplitRole] = {}
    group_owner: dict[str, SplitRole] = {}
    for manifest in manifests:
        for item in manifest.items:
            previous_item = item_owner.get(item.item_id)
            if previous_item is not None and previous_item != manifest.role:
                raise IdentityError(
                    f"item {item.item_id} appears in both {previous_item.value} "
                    f"and {manifest.role.value}"
                )
            item_owner[item.item_id] = manifest.role
            previous_group = group_owner.get(item.group_id)
            if previous_group is not None and previous_group != manifest.role:
                raise IdentityError(
                    f"semantic group {item.group_id} appears in both "
                    f"{previous_group.value} and {manifest.role.value}"
                )
            group_owner[item.group_id] = manifest.role


class EvaluationFunnel:
    """In-memory, immutable-evidence scheduler for candidate evaluation.

    The state is keyed only by content identities. Replaying the exact same
    observation is a cache hit; submitting different evidence to an occupied
    slot fails closed instead of silently replacing it.
    """

    def __init__(self) -> None:
        self._states: dict[str, _CandidateState] = {}
        self._cache_hits = 0
        self._cache_misses = 0

    def register_candidate(
        self,
        *,
        artifact_sha256: str,
        protocol: FunnelProtocol,
        geometry_split: SplitManifest,
        selection_split: SplitManifest,
        public_test_split: SplitManifest,
    ) -> RegistrationResult:
        _checked_sha256(artifact_sha256, field_name="artifact_sha256")
        if geometry_split.role is not SplitRole.TRAIN_GEOMETRY:
            raise IdentityError("geometry_split must have role train-geometry")
        if selection_split.role is not SplitRole.VALIDATION_SEARCH:
            raise IdentityError(
                "candidate selection may only use validation-search; "
                "public-test is forbidden"
            )
        if public_test_split.role is not SplitRole.PUBLIC_TEST:
            raise IdentityError("public_test_split must have role public-test")
        _require_no_split_leakage(
            (geometry_split, selection_split, public_test_split)
        )
        candidate = CandidateKey(
            artifact_sha256=artifact_sha256,
            protocol_sha256=protocol.sha256,
            geometry_split_sha256=geometry_split.sha256,
            search_split_sha256=selection_split.sha256,
            public_test_split_sha256=public_test_split.sha256,
        )
        existing = self._states.get(candidate.sha256)
        if existing is not None:
            if (
                existing.candidate != candidate
                or existing.protocol != protocol
                or existing.search_item_ids != selection_split.item_ids
            ):
                raise CacheConflictError("candidate hash collision with different inputs")
            self._cache_hits += 1
            return RegistrationResult(
                candidate.sha256,
                True,
                self._next_stage(existing),
                candidate,
            )
        self._states[candidate.sha256] = _CandidateState(
            candidate=candidate,
            protocol=protocol,
            search_item_ids=selection_split.item_ids,
        )
        self._cache_misses += 1
        return RegistrationResult(
            candidate.sha256,
            False,
            FunnelStage.REFUSAL,
            candidate,
        )

    def record_refusal(
        self,
        candidate: CandidateKey | str,
        observation: RefusalObservation,
    ) -> FunnelResult:
        state = self._state(candidate)
        self._require_search_item(state, observation.item_id)
        previous = state.refusals.get(observation.item_id)
        if previous is not None:
            return self._replay_or_conflict(state, previous, observation, "refusal")
        self._require_stage(state, FunnelStage.REFUSAL)
        state.refusals[observation.item_id] = observation
        self._cache_misses += 1
        return self._result(state, cache_hit=False)

    def record_kl(
        self,
        candidate: CandidateKey | str,
        observation: KLObservation,
    ) -> FunnelResult:
        state = self._state(candidate)
        self._require_search_item(state, observation.item_id)
        previous = state.kl_rows.get(observation.item_id)
        if previous is not None:
            return self._replay_or_conflict(state, previous, observation, "KL")
        self._require_stage(state, FunnelStage.KL)
        state.kl_rows[observation.item_id] = observation
        self._cache_misses += 1
        return self._result(state, cache_hit=False)

    def record_capability(
        self,
        candidate: CandidateKey | str,
        observation: CapabilityObservation,
    ) -> FunnelResult:
        state = self._state(candidate)
        if state.capability is not None:
            return self._replay_or_conflict(
                state,
                state.capability,
                observation,
                "capability",
            )
        self._require_stage(state, FunnelStage.CAPABILITY)
        state.capability = observation
        self._cache_misses += 1
        return self._result(state, cache_hit=False)

    def record_public_report(
        self,
        candidate: CandidateKey | str,
        observation: PublicReportObservation,
    ) -> FunnelResult:
        state = self._state(candidate)
        if state.public_report is not None:
            return self._replay_or_conflict(
                state,
                state.public_report,
                observation,
                "public report",
            )
        self._require_stage(state, FunnelStage.PUBLIC_REPORT)
        state.public_report = observation
        self._cache_misses += 1
        return self._result(state, cache_hit=False)

    def next_stage(self, candidate: CandidateKey | str) -> FunnelStage:
        return self._next_stage(self._state(candidate))

    def summary(self, candidate: CandidateKey | str) -> CandidateSummary:
        state = self._state(candidate)
        count = len(state.search_item_ids)
        refusal_count = self._semantic_refusal_count(state)
        kl_sum = math.fsum(row.value for row in state.kl_rows.values())
        kl_complete = len(state.kl_rows) == count
        capability = state.capability
        return CandidateSummary(
            candidate=state.candidate,
            next_stage=self._next_stage(state),
            refusal_evaluated=len(state.refusals),
            refusal_total=count,
            lexical_proxy_hits=sum(
                observation.lexical_proxy_hit
                for observation in state.refusals.values()
            ),
            semantic_refusals=refusal_count,
            kl_evaluated=len(state.kl_rows),
            kl_total=count,
            kl_sum=kl_sum,
            kl_lower_bound=kl_sum / count,
            mean_kl=kl_sum / count if kl_complete else None,
            capability_passed=None if capability is None else capability.passed,
            public_reported=state.public_report is not None,
        )

    @property
    def cache_stats(self) -> CacheStats:
        return CacheStats(self._cache_hits, self._cache_misses)

    def frontier(
        self,
        reference: CandidateKey | str,
    ) -> tuple[FrontierPoint, ...]:
        """Return the exact refusal/KL Pareto frontier for one matched protocol.

        Candidates are compared only when their protocol and validation-search
        split hashes match the reference. Partial KL runs are excluded because
        their final mean is not known.
        """

        reference_state = self._state(reference)
        points: list[FrontierPoint] = []
        for state in self._states.values():
            if (
                state.candidate.protocol_sha256
                != reference_state.candidate.protocol_sha256
                or state.candidate.search_split_sha256
                != reference_state.candidate.search_split_sha256
            ):
                continue
            if len(state.refusals) != len(state.search_item_ids):
                continue
            if len(state.kl_rows) != len(state.search_item_ids):
                continue
            points.append(
                FrontierPoint(
                    candidate_sha256=state.candidate.sha256,
                    artifact_sha256=state.candidate.artifact_sha256,
                    semantic_refusals=self._semantic_refusal_count(state),
                    mean_kl=math.fsum(
                        observation.value for observation in state.kl_rows.values()
                    )
                    / len(state.search_item_ids),
                )
            )
        nondominated = [
            point
            for point in points
            if not any(
                other.candidate_sha256 != point.candidate_sha256
                and other.semantic_refusals <= point.semantic_refusals
                and other.mean_kl <= point.mean_kl
                and (
                    other.semantic_refusals < point.semantic_refusals
                    or other.mean_kl < point.mean_kl
                )
                for other in points
            )
        ]
        return tuple(
            sorted(
                nondominated,
                key=lambda point: (
                    point.semantic_refusals,
                    point.mean_kl,
                    point.candidate_sha256,
                ),
            )
        )

    def _state(self, candidate: CandidateKey | str) -> _CandidateState:
        candidate_sha256 = (
            candidate.sha256 if isinstance(candidate, CandidateKey) else candidate
        )
        _checked_sha256(candidate_sha256, field_name="candidate_sha256")
        try:
            state = self._states[candidate_sha256]
        except KeyError as error:
            raise IdentityError(f"unknown candidate: {candidate_sha256}") from error
        if isinstance(candidate, CandidateKey) and candidate != state.candidate:
            raise IdentityError("candidate object does not match registered identity")
        return state

    @staticmethod
    def _require_search_item(state: _CandidateState, item_id: str) -> None:
        if item_id not in state.search_item_ids:
            raise IdentityError(
                f"item {item_id!r} is not in validation-search split "
                f"{state.candidate.search_split_sha256}"
            )

    @staticmethod
    def _semantic_refusal_count(state: _CandidateState) -> int:
        failure_labels = set(state.protocol.refusal_labels)
        return sum(
            observation.semantic_label in failure_labels
            for observation in state.refusals.values()
        )

    def _next_stage(self, state: _CandidateState) -> FunnelStage:
        if state.public_report is not None:
            return FunnelStage.COMPLETE
        refusal_count = self._semantic_refusal_count(state)
        if refusal_count > state.protocol.refusal_cap:
            return FunnelStage.REJECTED
        expected_count = len(state.search_item_ids)
        if len(state.refusals) < expected_count:
            return FunnelStage.REFUSAL
        kl_sum = math.fsum(observation.value for observation in state.kl_rows.values())
        # KL is non-negative. Even with zero on every remaining row, this is the
        # smallest possible final mean, so this rejection is exact.
        if kl_sum / expected_count > state.protocol.kl_cap:
            return FunnelStage.REJECTED
        if len(state.kl_rows) < expected_count:
            return FunnelStage.KL
        if state.capability is None:
            return FunnelStage.CAPABILITY
        if not state.capability.passed:
            return FunnelStage.REJECTED
        return FunnelStage.PUBLIC_REPORT

    def _require_stage(self, state: _CandidateState, expected: FunnelStage) -> None:
        actual = self._next_stage(state)
        if actual is not expected:
            raise StageOrderError(
                f"cannot record {expected.value} evidence while next stage is "
                f"{actual.value}"
            )

    def _replay_or_conflict(
        self,
        state: _CandidateState,
        previous: object,
        observation: object,
        evidence_kind: str,
    ) -> FunnelResult:
        if previous != observation:
            raise CacheConflictError(
                f"immutable {evidence_kind} cache slot already contains "
                "different evidence"
            )
        self._cache_hits += 1
        return self._result(state, cache_hit=True)

    def _result(self, state: _CandidateState, *, cache_hit: bool) -> FunnelResult:
        return FunnelResult(
            state.candidate.sha256,
            cache_hit,
            self._next_stage(state),
        )


__all__ = [
    "CacheConflictError",
    "CacheStats",
    "CandidateKey",
    "CandidateSummary",
    "CapabilityObservation",
    "EvaluationFunnel",
    "FrontierPoint",
    "FunnelError",
    "FunnelProtocol",
    "FunnelResult",
    "FunnelStage",
    "IdentityError",
    "KLObservation",
    "PublicReportObservation",
    "RefusalObservation",
    "RegistrationResult",
    "SemanticLabel",
    "SplitItem",
    "SplitManifest",
    "SplitRole",
    "StageOrderError",
    "VerdictSource",
]
