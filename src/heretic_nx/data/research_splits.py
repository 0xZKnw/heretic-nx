"""Frozen, content-addressed splits for research and public evaluation.

The generic hash-bucket assignment in :mod:`heretic_nx.data.splits` is kept
small on purpose.  This module binds those assignments to concrete dataset
rows so experiment scripts cannot silently swap the public test suite into a
geometry or candidate-selection phase.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

from heretic_nx.hashing import sha256_json

from .splits import SplitAssignment, SplitName, assign_split, validate_no_leakage


ResearchPurpose = Literal["geometry", "selection", "public-report"]

SPLIT_MANIFEST_SCHEMA = "heretic-nx-research-split-v1"
REFUSAL_NORMALIZER_V1 = "lowercase-apostrophe-whitespace-v1"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PURPOSE_ROLE: dict[ResearchPurpose, SplitName] = {
    "geometry": "train-geometry",
    "selection": "validation-search",
    "public-report": "public-test",
}
_PURPOSE_SOURCE: dict[ResearchPurpose, str] = {
    "geometry": "train",
    "selection": "train",
    "public-report": "test",
}


def _require_sha256(value: str, field_name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA-256")
    return value


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: frozenset[str],
    field_name: str,
) -> None:
    actual = frozenset(value)
    if actual != expected:
        raise ValueError(
            f"{field_name} keys differ: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _row_identity(
    text: str,
    *,
    dataset_id: str,
    revision: str,
    source_split: str,
    source_index: int,
) -> tuple[str, str, str]:
    prompt_sha256 = sha256_json({"text": text})
    normalized_group_text = " ".join(
        unicodedata.normalize("NFKC", text)
        .casefold()
        .replace("’", "'")
        .split()
    )
    group_id = sha256_json(
        {
            "normalized_text": normalized_group_text,
            "policy": "nfkc-casefold-apostrophe-whitespace-v1",
        }
    )
    item_id = sha256_json(
        {
            "dataset_id": dataset_id,
            "prompt_sha256": prompt_sha256,
            "revision": revision,
            "source_index": source_index,
            "source_split": source_split,
        }
    )
    return item_id, group_id, prompt_sha256


def refusal_marker_rule_sha256(
    markers: Sequence[str],
    *,
    normalizer: str = REFUSAL_NORMALIZER_V1,
) -> str:
    """Hash the exact normalization and marker rule used by an evaluator."""

    if normalizer != REFUSAL_NORMALIZER_V1:
        raise ValueError(f"unsupported refusal normalizer: {normalizer!r}")
    marker_list = list(markers)
    if (
        not marker_list
        or not all(isinstance(marker, str) and marker for marker in marker_list)
        or len(marker_list) != len(set(marker_list))
    ):
        raise ValueError("refusal markers must be unique non-empty strings")
    return sha256_json(
        {
            "normalizer": normalizer,
            "refusal_markers": marker_list,
        }
    )


def _marker_hits(
    responses: Sequence[str],
    markers: Sequence[str],
    *,
    normalizer: str,
) -> list[int]:
    if normalizer != REFUSAL_NORMALIZER_V1:
        raise ValueError(f"unsupported refusal normalizer: {normalizer!r}")
    return [
        int(
            any(
                marker in " ".join(response.lower().replace("’", "'").split())
                for marker in markers
            )
        )
        for response in responses
    ]


def _public_assignment_hash(
    *,
    item_id: str,
    group_id: str,
    seed: int,
) -> str:
    return sha256_json(
        {
            "group_id": group_id,
            "item_id": item_id,
            "policy": "reserved-dataset-test-v1",
            "seed": seed,
            "split": "public-test",
        }
    )


@dataclass(frozen=True, slots=True)
class ResearchSplitRow:
    """One source row whose identity includes both location and text."""

    item_id: str
    group_id: str
    prompt_sha256: str
    source_index: int
    assignment_hash: str

    def __post_init__(self) -> None:
        _require_sha256(self.item_id, "item_id")
        _require_sha256(self.group_id, "group_id")
        _require_sha256(self.prompt_sha256, "prompt_sha256")
        _require_sha256(self.assignment_hash, "assignment_hash")
        if (
            isinstance(self.source_index, bool)
            or not isinstance(self.source_index, int)
            or self.source_index < 0
        ):
            raise ValueError("source_index must be a non-negative integer")

    def to_dict(self) -> dict[str, object]:
        return {
            "assignment_hash": self.assignment_hash,
            "group_id": self.group_id,
            "item_id": self.item_id,
            "prompt_sha256": self.prompt_sha256,
            "source_index": self.source_index,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ResearchSplitRow:
        if not isinstance(value, Mapping):
            raise ValueError("split row must be an object")
        _require_exact_keys(
            value,
            frozenset(
                {
                    "assignment_hash",
                    "group_id",
                    "item_id",
                    "prompt_sha256",
                    "source_index",
                }
            ),
            "split row",
        )
        try:
            return cls(
                item_id=value["item_id"],
                group_id=value["group_id"],
                prompt_sha256=value["prompt_sha256"],
                source_index=value["source_index"],
                assignment_hash=value["assignment_hash"],
            )
        except KeyError as error:
            raise ValueError(f"split row is missing {error.args[0]}") from error


@dataclass(frozen=True, slots=True)
class ResearchSplitManifest:
    """Immutable binding between an experiment phase and concrete rows."""

    purpose: ResearchPurpose
    role: SplitName
    dataset_id: str
    revision: str
    source_split: str
    seed: int
    pool_size: int
    rows: tuple[ResearchSplitRow, ...]

    def __post_init__(self) -> None:
        if self.purpose not in _PURPOSE_ROLE:
            raise ValueError(f"unsupported research purpose: {self.purpose!r}")
        if self.role != _PURPOSE_ROLE[self.purpose]:
            raise ValueError(
                f"purpose {self.purpose} requires role {_PURPOSE_ROLE[self.purpose]}"
            )
        if self.source_split != _PURPOSE_SOURCE[self.purpose]:
            raise ValueError(
                f"purpose {self.purpose} requires dataset split "
                f"{_PURPOSE_SOURCE[self.purpose]!r}, not {self.source_split!r}"
            )
        if not isinstance(self.dataset_id, str) or not self.dataset_id:
            raise ValueError("dataset_id cannot be empty")
        if not isinstance(self.revision, str) or not self.revision:
            raise ValueError("dataset revision cannot be empty")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        if (
            isinstance(self.pool_size, bool)
            or not isinstance(self.pool_size, int)
            or self.pool_size <= 0
        ):
            raise ValueError("pool_size must be a positive integer")
        rows = tuple(self.rows)
        if not rows:
            raise ValueError("research split manifest cannot be empty")
        if not all(isinstance(row, ResearchSplitRow) for row in rows):
            raise ValueError("manifest rows must be ResearchSplitRow instances")
        indices = [row.source_index for row in rows]
        if len(indices) != len(set(indices)):
            raise ValueError("manifest source indices must be unique")
        if max(indices) >= self.pool_size:
            raise ValueError("manifest source index is outside its frozen pool")
        item_ids = [row.item_id for row in rows]
        if len(item_ids) != len(set(item_ids)):
            raise ValueError("manifest item IDs must be unique")

        assignments = []
        for row in rows:
            expected_item_id = sha256_json(
                {
                    "dataset_id": self.dataset_id,
                    "prompt_sha256": row.prompt_sha256,
                    "revision": self.revision,
                    "source_index": row.source_index,
                    "source_split": self.source_split,
                }
            )
            if row.item_id != expected_item_id:
                raise ValueError(
                    f"row {row.source_index} has an invalid item identity"
                )
            if self.purpose == "public-report":
                expected_hash = _public_assignment_hash(
                    item_id=row.item_id,
                    group_id=row.group_id,
                    seed=self.seed,
                )
            else:
                expected = assign_split(
                    row.item_id,
                    seed=self.seed,
                    group_id=row.group_id,
                )
                if expected.split != self.role:
                    raise ValueError(
                        f"row {row.source_index} hashes to {expected.split}, "
                        f"not {self.role}"
                    )
                expected_hash = expected.assignment_hash
            if row.assignment_hash != expected_hash:
                raise ValueError(
                    f"row {row.source_index} has an invalid assignment hash"
                )
            assignments.append(
                SplitAssignment(
                    row.item_id,
                    self.role,
                    row.assignment_hash,
                    row.group_id,
                )
            )
        validate_no_leakage(assignments)
        object.__setattr__(self, "rows", rows)

    @property
    def sha256(self) -> str:
        return sha256_json(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset": {"id": self.dataset_id, "revision": self.revision},
            "pool_size": self.pool_size,
            "purpose": self.purpose,
            "role": self.role,
            "rows": [row.to_dict() for row in self.rows],
            "schema_version": SPLIT_MANIFEST_SCHEMA,
            "seed": self.seed,
            "source_split": self.source_split,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ResearchSplitManifest:
        if not isinstance(value, Mapping):
            raise ValueError("split manifest must be an object")
        if value.get("schema_version") != SPLIT_MANIFEST_SCHEMA:
            raise ValueError("unsupported or missing research split schema")
        dataset = value.get("dataset")
        if not isinstance(dataset, Mapping):
            raise ValueError("split manifest dataset must be an object")
        _require_exact_keys(
            value,
            frozenset(
                {
                    "dataset",
                    "pool_size",
                    "purpose",
                    "role",
                    "rows",
                    "schema_version",
                    "seed",
                    "source_split",
                }
            ),
            "split manifest",
        )
        _require_exact_keys(
            dataset,
            frozenset({"id", "revision"}),
            "split manifest dataset",
        )
        rows = value.get("rows")
        if not isinstance(rows, list):
            raise ValueError("split manifest rows must be a list")
        try:
            return cls(
                purpose=value["purpose"],
                role=value["role"],
                dataset_id=dataset["id"],
                revision=dataset["revision"],
                source_split=value["source_split"],
                seed=value["seed"],
                pool_size=value["pool_size"],
                rows=tuple(ResearchSplitRow.from_dict(row) for row in rows),
            )
        except KeyError as error:
            raise ValueError(f"split manifest is missing {error.args[0]}") from error


def build_research_split(
    texts: Sequence[str],
    *,
    purpose: ResearchPurpose,
    dataset_id: str,
    revision: str,
    source_split: str,
    seed: int,
    count: int | None = None,
) -> ResearchSplitManifest:
    """Assign and freeze concrete source rows for one experiment phase.

    Geometry and selection rows are hash-partitioned from the dataset's
    training split.  The dataset's test split is reserved wholesale for public
    reporting and can never be requested under a research purpose.
    """

    if purpose not in _PURPOSE_ROLE:
        raise ValueError(f"unsupported research purpose: {purpose!r}")
    if source_split != _PURPOSE_SOURCE[purpose]:
        raise ValueError(
            f"purpose {purpose} cannot read dataset split {source_split!r}"
        )
    if count is not None and (
        isinstance(count, bool) or not isinstance(count, int) or count <= 0
    ):
        raise ValueError("count must be a positive integer or None")
    if not texts:
        raise ValueError("cannot freeze an empty dataset pool")

    role = _PURPOSE_ROLE[purpose]
    eligible: list[ResearchSplitRow] = []
    for source_index, text in enumerate(texts):
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"dataset row {source_index} has empty text")
        item_id, group_id, prompt_sha256 = _row_identity(
            text,
            dataset_id=dataset_id,
            revision=revision,
            source_split=source_split,
            source_index=source_index,
        )
        if purpose == "public-report":
            assignment_hash = _public_assignment_hash(
                item_id=item_id,
                group_id=group_id,
                seed=seed,
            )
        else:
            assignment = assign_split(
                item_id,
                seed=seed,
                group_id=group_id,
            )
            if assignment.split != role:
                continue
            assignment_hash = assignment.assignment_hash
        eligible.append(
            ResearchSplitRow(
                item_id=item_id,
                group_id=group_id,
                prompt_sha256=prompt_sha256,
                source_index=source_index,
                assignment_hash=assignment_hash,
            )
        )
    if count is not None:
        if len(eligible) < count:
            raise ValueError(
                f"requested {count} {role} rows but frozen pool only has "
                f"{len(eligible)}"
            )
        eligible = eligible[:count]
    return ResearchSplitManifest(
        purpose=purpose,
        role=role,
        dataset_id=dataset_id,
        revision=revision,
        source_split=source_split,
        seed=seed,
        pool_size=len(texts),
        rows=tuple(eligible),
    )


def subset_research_split(
    manifest: ResearchSplitManifest,
    positions: Sequence[int],
) -> ResearchSplitManifest:
    if not positions:
        raise ValueError("research split subset cannot be empty")
    if any(
        isinstance(position, bool) or not isinstance(position, int)
        for position in positions
    ):
        raise ValueError("research split positions must be integers")
    if len(set(positions)) != len(positions):
        raise ValueError("research split positions must be unique")
    if min(positions) < 0 or max(positions) >= len(manifest.rows):
        raise ValueError("research split position is out of range")
    return ResearchSplitManifest(
        purpose=manifest.purpose,
        role=manifest.role,
        dataset_id=manifest.dataset_id,
        revision=manifest.revision,
        source_split=manifest.source_split,
        seed=manifest.seed,
        pool_size=manifest.pool_size,
        rows=tuple(manifest.rows[position] for position in positions),
    )


def verify_manifest_texts(
    manifest: ResearchSplitManifest,
    texts: Sequence[str],
) -> list[str]:
    """Return selected texts only after verifying the frozen row identities."""

    if len(texts) != manifest.pool_size:
        raise ValueError(
            f"dataset pool changed size: {len(texts)} != {manifest.pool_size}"
        )
    selected = []
    for row in manifest.rows:
        text = texts[row.source_index]
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"dataset row {row.source_index} has empty text")
        identity = _row_identity(
            text,
            dataset_id=manifest.dataset_id,
            revision=manifest.revision,
            source_split=manifest.source_split,
            source_index=row.source_index,
        )
        if identity != (row.item_id, row.group_id, row.prompt_sha256):
            raise ValueError(f"dataset row {row.source_index} changed after freeze")
        selected.append(text)
    return selected


def assert_research_splits_disjoint(
    *manifests: ResearchSplitManifest,
) -> None:
    """Reject item or normalized-content reuse across different phase roles."""

    item_roles: dict[str, SplitName] = {}
    group_roles: dict[str, SplitName] = {}
    for manifest in manifests:
        for row in manifest.rows:
            previous_item = item_roles.get(row.item_id)
            if previous_item is not None and previous_item != manifest.role:
                raise ValueError(
                    f"item {row.item_id} appears in {previous_item} and "
                    f"{manifest.role}"
                )
            item_roles[row.item_id] = manifest.role
            previous_group = group_roles.get(row.group_id)
            if previous_group is not None and previous_group != manifest.role:
                raise ValueError(
                    f"normalized-content group {row.group_id} appears in {previous_group} "
                    f"and {manifest.role}"
                )
            group_roles[row.group_id] = manifest.role


def manifest_from_report(
    report: Mapping[str, Any],
    *,
    expected_purpose: ResearchPurpose,
    expected_artifact_sha256: str,
    expected_schema_version: str,
    expected_protocol_schema_version: str,
    expected_marker_rule_sha256: str,
    expected_full_manifest_sha256: str,
) -> ResearchSplitManifest:
    """Validate a report against an independently pinned experiment contract."""

    if not isinstance(report, Mapping):
        raise ValueError("evaluation report must be an object")
    schema_version = report.get("schema_version")
    if not isinstance(schema_version, str) or not schema_version:
        raise ValueError("evaluation report has no schema version")
    if schema_version != expected_schema_version:
        raise ValueError(
            f"expected report schema {expected_schema_version!r}, "
            f"got {schema_version!r}"
        )
    manifest_value = report.get("split_manifest")
    if not isinstance(manifest_value, Mapping):
        raise ValueError("evaluation report has no frozen split manifest")
    manifest = ResearchSplitManifest.from_dict(manifest_value)
    if manifest.purpose != expected_purpose:
        raise ValueError(
            f"expected {expected_purpose} report, got {manifest.purpose}"
        )
    digest = report.get("split_manifest_sha256")
    if digest != manifest.sha256:
        raise ValueError("evaluation report split manifest hash mismatch")
    full_manifest_value = report.get("full_split_manifest")
    if not isinstance(full_manifest_value, Mapping):
        raise ValueError("evaluation report has no full split manifest")
    full_manifest = ResearchSplitManifest.from_dict(full_manifest_value)
    full_digest = report.get("full_split_manifest_sha256")
    if full_digest != full_manifest.sha256:
        raise ValueError("evaluation report full split manifest hash mismatch")
    if full_digest != _require_sha256(
        expected_full_manifest_sha256,
        "expected_full_manifest_sha256",
    ):
        raise ValueError("evaluation report uses an unexpected full split")
    if full_manifest != manifest:
        raise ValueError("evaluation report only covers a subset of the full split")
    if report.get("complete") is not True:
        raise ValueError("evaluation report is incomplete")
    if report.get("suite_complete") is not True:
        raise ValueError("evaluation report does not certify the full suite")
    count = report.get("count")
    expected_count = report.get("expected_count")
    responses = report.get("responses")
    marker_hits = report.get("marker_hits")
    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or count != len(manifest.rows)
        or isinstance(expected_count, bool)
        or not isinstance(expected_count, int)
        or expected_count != len(full_manifest.rows)
        or not isinstance(responses, list)
        or not isinstance(marker_hits, list)
        or len(responses) != count
        or len(marker_hits) != count
        or not all(isinstance(response, str) for response in responses)
        or not all(
            not isinstance(hit, bool) and isinstance(hit, int) and hit in {0, 1}
            for hit in marker_hits
        )
    ):
        raise ValueError("evaluation report rows and evidence are misaligned")
    if report.get("refusal_markers") != sum(marker_hits):
        raise ValueError("evaluation report refusal total is inconsistent")
    response_sha256 = sha256_json(responses)
    if report.get("response_sha256") != response_sha256:
        raise ValueError("evaluation report response hash mismatch")
    evidence_sha256 = sha256_json(
        {"marker_hits": marker_hits, "responses": responses}
    )
    if report.get("evidence_sha256") != evidence_sha256:
        raise ValueError("evaluation report evidence hash mismatch")
    artifact_sha256 = report.get("artifact_sha256")
    _require_sha256(artifact_sha256, "artifact_sha256")
    runtime_model = report.get("runtime_model")
    if (
        report.get("artifact_attested") is not True
        or not isinstance(runtime_model, Mapping)
        or runtime_model.get("artifact_sha256") != artifact_sha256
        or runtime_model.get("model_alias") != report.get("model")
    ):
        raise ValueError("evaluation report model artifact is not attested")
    if artifact_sha256 != _require_sha256(
        expected_artifact_sha256,
        "expected_artifact_sha256",
    ):
        raise ValueError("evaluation report belongs to another model artifact")
    dataset = report.get("dataset")
    if not isinstance(dataset, Mapping):
        raise ValueError("evaluation report dataset must be an object")
    _require_exact_keys(
        dataset,
        frozenset({"id", "revision", "source_split"}),
        "evaluation report dataset",
    )
    if dict(dataset) != {
        "id": manifest.dataset_id,
        "revision": manifest.revision,
        "source_split": manifest.source_split,
    }:
        raise ValueError("evaluation report dataset does not match its manifest")
    protocol = report.get("protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError("evaluation protocol must be an object")
    protocol_schema_version = protocol.get("schema_version")
    if not isinstance(protocol_schema_version, str) or not protocol_schema_version:
        raise ValueError("evaluation protocol has no schema version")
    if protocol_schema_version != expected_protocol_schema_version:
        raise ValueError(
            f"expected protocol schema {expected_protocol_schema_version!r}, "
            f"got {protocol_schema_version!r}"
        )
    if (
        protocol.get("split_manifest_sha256") != manifest.sha256
        or protocol.get("full_split_manifest_sha256") != full_manifest.sha256
        or protocol.get("purpose") != expected_purpose
    ):
        raise ValueError("evaluation protocol is not bound to its split manifest")
    full_split_count = protocol.get("full_split_count")
    if (
        isinstance(full_split_count, bool)
        or not isinstance(full_split_count, int)
        or full_split_count != len(full_manifest.rows)
    ):
        raise ValueError("evaluation protocol has an invalid full split count")
    expected_indices = [row.source_index + 1 for row in manifest.rows]
    if protocol.get("source_indices_one_based") != expected_indices:
        raise ValueError("evaluation protocol source indices do not match")
    _require_sha256(
        protocol.get("prompt_tokens_sha256"),
        "protocol.prompt_tokens_sha256",
    )
    if (
        protocol.get("model") != report.get("model")
        or protocol.get("artifact_sha256") != artifact_sha256
        or protocol.get("artifact_attested") is not True
    ):
        raise ValueError("evaluation protocol is not bound to its model artifact")
    refusal_markers = protocol.get("refusal_markers")
    if not isinstance(refusal_markers, list):
        raise ValueError("evaluation protocol refusal markers must be a list")
    normalizer = protocol.get("refusal_normalizer")
    rule_sha256 = refusal_marker_rule_sha256(
        refusal_markers,
        normalizer=normalizer,
    )
    if protocol.get("marker_rule_sha256") != rule_sha256:
        raise ValueError("evaluation protocol marker rule hash mismatch")
    if rule_sha256 != _require_sha256(
        expected_marker_rule_sha256,
        "expected_marker_rule_sha256",
    ):
        raise ValueError("evaluation report uses an unexpected marker rule")
    if marker_hits != _marker_hits(
        responses,
        refusal_markers,
        normalizer=normalizer,
    ):
        raise ValueError("evaluation report marker hits do not match responses")
    return manifest


__all__ = [
    "REFUSAL_NORMALIZER_V1",
    "ResearchPurpose",
    "ResearchSplitManifest",
    "ResearchSplitRow",
    "SPLIT_MANIFEST_SCHEMA",
    "assert_research_splits_disjoint",
    "build_research_split",
    "manifest_from_report",
    "refusal_marker_rule_sha256",
    "subset_research_split",
    "verify_manifest_texts",
]
