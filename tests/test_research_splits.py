from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from heretic_nx.data.research_splits import (
    REFUSAL_NORMALIZER_V1,
    ResearchSplitManifest,
    assert_research_splits_disjoint,
    build_research_split,
    manifest_from_report,
    refusal_marker_rule_sha256,
    subset_research_split,
    verify_manifest_texts,
)
from heretic_nx.hashing import sha256_json


def pool(size: int = 300) -> list[str]:
    return [f"stable prompt {index} with unique content" for index in range(size)]


def test_train_geometry_and_validation_are_deterministic_and_disjoint() -> None:
    texts = pool()
    geometry = build_research_split(
        texts,
        purpose="geometry",
        dataset_id="owner/data",
        revision="abc123",
        source_split="train",
        seed=7300,
    )
    selection = build_research_split(
        texts,
        purpose="selection",
        dataset_id="owner/data",
        revision="abc123",
        source_split="train",
        seed=7300,
    )
    assert geometry.role == "train-geometry"
    assert selection.role == "validation-search"
    assert set(row.source_index for row in geometry.rows).isdisjoint(
        row.source_index for row in selection.rows
    )
    assert_research_splits_disjoint(geometry, selection)
    rebuilt = ResearchSplitManifest.from_dict(geometry.to_dict())
    assert rebuilt == geometry
    assert rebuilt.sha256 == geometry.sha256
    assert verify_manifest_texts(rebuilt, texts) == [
        texts[row.source_index] for row in rebuilt.rows
    ]


def test_source_partition_policy_fails_closed() -> None:
    texts = pool(20)
    with pytest.raises(ValueError, match="cannot read dataset split"):
        build_research_split(
            texts,
            purpose="geometry",
            dataset_id="owner/data",
            revision="abc123",
            source_split="test",
            seed=1,
        )
    with pytest.raises(ValueError, match="cannot read dataset split"):
        build_research_split(
            texts,
            purpose="public-report",
            dataset_id="owner/data",
            revision="abc123",
            source_split="train",
            seed=1,
        )


def test_manifest_detects_dataset_and_manifest_tampering() -> None:
    texts = pool(80)
    manifest = build_research_split(
        texts,
        purpose="geometry",
        dataset_id="owner/data",
        revision="abc123",
        source_split="train",
        seed=17,
        count=8,
    )
    changed = list(texts)
    changed[manifest.rows[0].source_index] += " changed"
    with pytest.raises(ValueError, match="changed after freeze"):
        verify_manifest_texts(manifest, changed)

    payload = manifest.to_dict()
    payload["rows"][0]["assignment_hash"] = "0" * 64
    with pytest.raises(ValueError, match="invalid assignment hash"):
        ResearchSplitManifest.from_dict(payload)

    identity_payload = manifest.to_dict()
    identity_payload["rows"][0]["item_id"] = "f" * 64
    with pytest.raises(ValueError, match="invalid item identity"):
        ResearchSplitManifest.from_dict(identity_payload)

    group_payload = manifest.to_dict()
    group_payload["rows"][0]["group_id"] = "f" * 64
    with pytest.raises(ValueError, match="hashes to|invalid assignment hash"):
        ResearchSplitManifest.from_dict(group_payload)

    extra_payload = manifest.to_dict()
    extra_payload["unexpected"] = True
    with pytest.raises(ValueError, match="extra=.*unexpected"):
        ResearchSplitManifest.from_dict(extra_payload)


def test_semantic_duplicates_cannot_cross_research_and_public_roles() -> None:
    texts = pool(80)
    geometry = build_research_split(
        texts,
        purpose="geometry",
        dataset_id="owner/data",
        revision="abc123",
        source_split="train",
        seed=29,
        count=1,
    )
    duplicated = [
        "  "
        + texts[geometry.rows[0].source_index].upper().replace(" ", "   ")
        + "  "
    ]
    public = build_research_split(
        duplicated,
        purpose="public-report",
        dataset_id="owner/data",
        revision="abc123",
        source_split="test",
        seed=29,
    )
    with pytest.raises(ValueError, match="normalized-content group"):
        assert_research_splits_disjoint(geometry, public)


def valid_report(
    manifest: ResearchSplitManifest,
    *,
    artifact_sha256: str = "a" * 64,
) -> dict[str, object]:
    responses = ["ok"] * len(manifest.rows)
    marker_hits = [0] * len(manifest.rows)
    marker_rule_sha256 = refusal_marker_rule_sha256(["sorry", "cannot"])
    return {
        "schema_version": "test-refusal-v2",
        "artifact_sha256": artifact_sha256,
        "artifact_attested": True,
        "complete": True,
        "suite_complete": True,
        "count": len(manifest.rows),
        "expected_count": len(manifest.rows),
        "dataset": {
            "id": manifest.dataset_id,
            "revision": manifest.revision,
            "source_split": manifest.source_split,
        },
        "marker_hits": marker_hits,
        "protocol": {
            "schema_version": "test-protocol-v2",
            "artifact_attested": True,
            "artifact_sha256": artifact_sha256,
            "full_split_count": len(manifest.rows),
            "full_split_manifest_sha256": manifest.sha256,
            "marker_rule_sha256": marker_rule_sha256,
            "model": "candidate",
            "prompt_tokens_sha256": "b" * 64,
            "purpose": manifest.purpose,
            "refusal_markers": ["sorry", "cannot"],
            "refusal_normalizer": REFUSAL_NORMALIZER_V1,
            "source_indices_one_based": [
                row.source_index + 1 for row in manifest.rows
            ],
            "split_manifest_sha256": manifest.sha256,
        },
        "refusal_markers": 0,
        "responses": responses,
        "response_sha256": sha256_json(responses),
        "runtime_model": {
            "artifact_sha256": artifact_sha256,
            "model_alias": "candidate",
        },
        "model": "candidate",
        "split_manifest": manifest.to_dict(),
        "split_manifest_sha256": manifest.sha256,
        "full_split_manifest": manifest.to_dict(),
        "full_split_manifest_sha256": manifest.sha256,
        "evidence_sha256": sha256_json(
            {"marker_hits": marker_hits, "responses": responses}
        ),
    }


def expected_report_contract(
    manifest: ResearchSplitManifest,
    *,
    artifact_sha256: str = "a" * 64,
) -> dict[str, str]:
    return {
        "expected_artifact_sha256": artifact_sha256,
        "expected_schema_version": "test-refusal-v2",
        "expected_protocol_schema_version": "test-protocol-v2",
        "expected_marker_rule_sha256": refusal_marker_rule_sha256(
            ["sorry", "cannot"]
        ),
        "expected_full_manifest_sha256": manifest.sha256,
    }


def test_evaluation_report_must_be_complete_and_manifest_bound() -> None:
    selection = build_research_split(
        pool(120),
        purpose="selection",
        dataset_id="owner/data",
        revision="abc123",
        source_split="train",
        seed=41,
        count=10,
    )
    report = valid_report(selection)
    marker_rule_sha256 = refusal_marker_rule_sha256(["sorry", "cannot"])
    assert (
        manifest_from_report(
            report,
            expected_purpose="selection",
            expected_artifact_sha256="a" * 64,
            expected_schema_version="test-refusal-v2",
            expected_protocol_schema_version="test-protocol-v2",
            expected_marker_rule_sha256=marker_rule_sha256,
            expected_full_manifest_sha256=selection.sha256,
        )
        == selection
    )
    report["complete"] = False
    with pytest.raises(ValueError, match="incomplete"):
        manifest_from_report(
            report,
            expected_purpose="selection",
            **expected_report_contract(selection),
        )
    report["complete"] = True
    report["split_manifest_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="hash mismatch"):
        manifest_from_report(
            report,
            expected_purpose="selection",
            **expected_report_contract(selection),
        )


def test_evaluation_subset_cannot_certify_the_full_suite() -> None:
    full = build_research_split(
        pool(120),
        purpose="selection",
        dataset_id="owner/data",
        revision="abc123",
        source_split="train",
        seed=73,
        count=10,
    )
    subset = subset_research_split(full, [0])
    report = valid_report(subset)
    report["full_split_manifest"] = full.to_dict()
    report["full_split_manifest_sha256"] = full.sha256
    protocol = report["protocol"]
    assert isinstance(protocol, dict)
    protocol["full_split_count"] = len(full.rows)
    protocol["full_split_manifest_sha256"] = full.sha256
    with pytest.raises(ValueError, match="only covers a subset"):
        manifest_from_report(
            report,
            expected_purpose="selection",
            **expected_report_contract(full),
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("response", "response hash mismatch"),
        ("evidence", "evidence hash mismatch"),
        ("marker-hit", "marker hits do not match responses"),
        ("marker-rule", "marker rule hash mismatch"),
        ("dataset", "dataset does not match"),
        ("indices", "source indices do not match"),
        ("purpose", "not bound to its split manifest"),
        ("schema", "expected report schema"),
        ("protocol-schema", "expected protocol schema"),
    ],
)
def test_evaluation_report_rejects_cross_wired_evidence(
    mutation: str,
    message: str,
) -> None:
    manifest = build_research_split(
        pool(100),
        purpose="selection",
        dataset_id="owner/data",
        revision="abc123",
        source_split="train",
        seed=79,
        count=4,
    )
    report = deepcopy(valid_report(manifest))
    protocol = report["protocol"]
    assert isinstance(protocol, dict)
    if mutation == "response":
        report["response_sha256"] = "c" * 64
    elif mutation == "evidence":
        report["evidence_sha256"] = "c" * 64
    elif mutation == "marker-hit":
        responses = report["responses"]
        assert isinstance(responses, list)
        responses[0] = "sorry, cannot do that"
        report["response_sha256"] = sha256_json(responses)
        marker_hits = report["marker_hits"]
        assert isinstance(marker_hits, list)
        report["evidence_sha256"] = sha256_json(
            {"marker_hits": marker_hits, "responses": responses}
        )
    elif mutation == "marker-rule":
        protocol["marker_rule_sha256"] = "c" * 64
    elif mutation == "dataset":
        dataset = report["dataset"]
        assert isinstance(dataset, dict)
        dataset["revision"] = "other"
    elif mutation == "indices":
        protocol["source_indices_one_based"] = [999] * len(manifest.rows)
    elif mutation == "purpose":
        protocol["purpose"] = "geometry"
    elif mutation == "schema":
        report["schema_version"] = "other-v1"
    elif mutation == "protocol-schema":
        protocol["schema_version"] = "other-v1"
    with pytest.raises(ValueError, match=message):
        manifest_from_report(
            report,
            expected_purpose="selection",
            **expected_report_contract(manifest),
        )


def test_evaluation_report_recomputes_refusal_hits() -> None:
    manifest = build_research_split(
        pool(100),
        purpose="selection",
        dataset_id="owner/data",
        revision="abc123",
        source_split="train",
        seed=83,
        count=2,
    )
    report = valid_report(manifest)
    responses = ["SORRY, I cannot do that.", "fine"]
    hits = [1, 0]
    report["responses"] = responses
    report["marker_hits"] = hits
    report["refusal_markers"] = 1
    report["response_sha256"] = sha256_json(responses)
    report["evidence_sha256"] = sha256_json(
        {"marker_hits": hits, "responses": responses}
    )
    assert (
        manifest_from_report(
            report,
            expected_purpose="selection",
            **expected_report_contract(manifest),
        )
        == manifest
    )


def test_weight_fitting_workflows_do_not_read_the_public_test_suite() -> None:
    root = Path(__file__).resolve().parents[1]
    ling_sites = (root / "experiments/ling3_tiny_native_dense_sites.py").read_text()
    teacher_inputs = (root / "experiments/lfm25_8b_a1b_teacher_inputs.py").read_text()
    assert 'split="test"' not in ling_sites
    assert 'split="test"' not in teacher_inputs
    assert 'range(104)' not in teacher_inputs
    assert 'base-full.json' not in teacher_inputs
    assert 'prime-ops8-b2-full.json' not in teacher_inputs
    assert 'expected_purpose="geometry"' in teacher_inputs
    assert 'expected_purpose="selection"' in teacher_inputs


def test_manifest_serialization_is_canonical_json_compatible() -> None:
    manifest = build_research_split(
        pool(50),
        purpose="public-report",
        dataset_id="owner/data",
        revision="abc123",
        source_split="test",
        seed=53,
        count=12,
    )
    assert ResearchSplitManifest.from_dict(
        json.loads(json.dumps(manifest.to_dict()))
    ).sha256 == manifest.sha256
