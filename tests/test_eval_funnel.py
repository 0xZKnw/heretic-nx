from __future__ import annotations

import math

import pytest

from heretic_nx.eval.funnel import (
    CacheConflictError,
    CapabilityObservation,
    EvaluationFunnel,
    FunnelProtocol,
    FunnelStage,
    IdentityError,
    KLObservation,
    PublicReportObservation,
    RefusalObservation,
    SemanticLabel,
    SplitItem,
    SplitManifest,
    SplitRole,
    StageOrderError,
    VerdictSource,
)


def sha(index: int) -> str:
    return f"{index:064x}"


def split(
    role: SplitRole,
    prefix: str,
    count: int,
    *,
    prompt_offset: int,
    group_prefix: str | None = None,
) -> SplitManifest:
    return SplitManifest(
        role,
        tuple(
            SplitItem(
                f"{prefix}-{index}",
                f"{group_prefix or prefix}-group-{index}",
                sha(prompt_offset + index),
            )
            for index in range(count)
        ),
    )


def protocol(*, refusal_cap: int = 1, kl_cap: float = 0.05, salt: int = 0) -> FunnelProtocol:
    return FunnelProtocol(
        runtime_sha256=sha(10 + salt),
        tokenizer_sha256=sha(20 + salt),
        chat_template_sha256=sha(30 + salt),
        generation_config_sha256=sha(40 + salt),
        lexical_rule_sha256=sha(50 + salt),
        semantic_rule_sha256=sha(60 + salt),
        kl_spec_sha256=sha(70 + salt),
        capability_spec_sha256=sha(80 + salt),
        refusal_cap=refusal_cap,
        kl_cap=kl_cap,
    )


@pytest.fixture
def manifests() -> tuple[SplitManifest, SplitManifest, SplitManifest]:
    return (
        split(SplitRole.TRAIN_GEOMETRY, "geometry", 2, prompt_offset=100),
        split(SplitRole.VALIDATION_SEARCH, "search", 3, prompt_offset=200),
        split(SplitRole.PUBLIC_TEST, "public", 2, prompt_offset=300),
    )


def register(
    funnel: EvaluationFunnel,
    manifests: tuple[SplitManifest, SplitManifest, SplitManifest],
    *,
    artifact: int,
    active_protocol: FunnelProtocol,
):
    geometry, search, public = manifests
    return funnel.register_candidate(
        artifact_sha256=sha(artifact),
        protocol=active_protocol,
        geometry_split=geometry,
        selection_split=search,
        public_test_split=public,
    ).candidate


def refusal(
    item_id: str,
    index: int,
    *,
    label: SemanticLabel = SemanticLabel.COMPLIANCE,
    lexical_hit: bool = False,
    source: VerdictSource = VerdictSource.DETERMINISTIC,
) -> RefusalObservation:
    return RefusalObservation(
        item_id=item_id,
        response_sha256=sha(1_000 + index),
        lexical_proxy_hit=lexical_hit,
        semantic_label=label,
        semantic_source=source,
        semantic_evidence_sha256=sha(2_000 + index),
    )


def kl(item_id: str, index: int, value: float) -> KLObservation:
    return KLObservation(item_id, value, sha(3_000 + index))


def complete_refusal(
    funnel: EvaluationFunnel,
    candidate,
    search: SplitManifest,
    *,
    refusal_indices: set[int] | None = None,
) -> None:
    refusal_indices = refusal_indices or set()
    for index, item in enumerate(search.items):
        label = (
            SemanticLabel.REFUSAL
            if index in refusal_indices
            else SemanticLabel.COMPLIANCE
        )
        funnel.record_refusal(candidate, refusal(item.item_id, index, label=label))


def complete_kl(
    funnel: EvaluationFunnel,
    candidate,
    search: SplitManifest,
    values: list[float],
) -> None:
    assert len(values) == len(search.items)
    for index, (item, value) in enumerate(zip(search.items, values)):
        funnel.record_kl(candidate, kl(item.item_id, index, value))


def test_manifests_and_protocols_are_content_addressed() -> None:
    original = split(SplitRole.VALIDATION_SEARCH, "row", 2, prompt_offset=10)
    replay = split(SplitRole.VALIDATION_SEARCH, "row", 2, prompt_offset=10)
    reordered = SplitManifest(original.role, tuple(reversed(original.items)))
    public = SplitManifest(SplitRole.PUBLIC_TEST, original.items)
    assert original.sha256 == replay.sha256
    assert original.sha256 != reordered.sha256
    assert original.sha256 != public.sha256
    assert protocol(refusal_cap=1).sha256 == protocol(refusal_cap=1).sha256
    assert protocol(refusal_cap=1).sha256 != protocol(refusal_cap=2).sha256
    assert protocol(kl_cap=0.05).sha256 != protocol(kl_cap=0.06).sha256


def test_split_manifest_rejects_duplicate_rows_and_bad_hashes() -> None:
    item = SplitItem("one", "group", sha(1))
    with pytest.raises(IdentityError, match="unique"):
        SplitManifest(SplitRole.VALIDATION_SEARCH, (item, item))
    with pytest.raises(IdentityError, match="SHA-256"):
        SplitItem("one", "group", "not-a-hash")
    with pytest.raises(IdentityError, match="empty"):
        SplitManifest(SplitRole.VALIDATION_SEARCH, ())


def test_selection_rejects_public_test_and_cross_split_leakage(manifests) -> None:
    geometry, search, public = manifests
    funnel = EvaluationFunnel()
    with pytest.raises(IdentityError, match="public-test is forbidden"):
        funnel.register_candidate(
            artifact_sha256=sha(900),
            protocol=protocol(),
            geometry_split=geometry,
            selection_split=public,
            public_test_split=public,
        )

    leaked_public = SplitManifest(
        SplitRole.PUBLIC_TEST,
        (SplitItem("other", search.items[0].group_id, sha(901)),),
    )
    with pytest.raises(IdentityError, match="semantic group"):
        funnel.register_candidate(
            artifact_sha256=sha(902),
            protocol=protocol(),
            geometry_split=geometry,
            selection_split=search,
            public_test_split=leaked_public,
        )


def test_full_refusal_then_kl_then_capability_then_public_report(manifests) -> None:
    _, search, _ = manifests
    funnel = EvaluationFunnel()
    candidate = register(
        funnel,
        manifests,
        artifact=500,
        active_protocol=protocol(refusal_cap=1, kl_cap=0.05),
    )
    assert funnel.next_stage(candidate) is FunnelStage.REFUSAL
    funnel.record_refusal(candidate, refusal(search.items[0].item_id, 0))
    funnel.record_refusal(candidate, refusal(search.items[1].item_id, 1))
    with pytest.raises(StageOrderError, match="next stage is refusal"):
        funnel.record_kl(candidate, kl(search.items[0].item_id, 0, 0.01))

    result = funnel.record_refusal(
        candidate,
        refusal(
            search.items[2].item_id,
            2,
            label=SemanticLabel.REFUSAL,
        ),
    )
    assert result.next_stage is FunnelStage.KL
    with pytest.raises(StageOrderError, match="next stage is kl"):
        funnel.record_capability(candidate, CapabilityObservation(True, sha(700)))

    complete_kl(funnel, candidate, search, [0.01, 0.02, 0.03])
    assert funnel.next_stage(candidate) is FunnelStage.CAPABILITY
    with pytest.raises(StageOrderError, match="next stage is capability"):
        funnel.record_public_report(candidate, PublicReportObservation(sha(701)))
    result = funnel.record_capability(
        candidate,
        CapabilityObservation(True, sha(702)),
    )
    assert result.next_stage is FunnelStage.PUBLIC_REPORT
    result = funnel.record_public_report(
        candidate,
        PublicReportObservation(sha(703)),
    )
    assert result.next_stage is FunnelStage.COMPLETE

    summary = funnel.summary(candidate)
    assert summary.refusal_evaluated == summary.refusal_total == 3
    assert summary.semantic_refusals == 1
    assert summary.kl_evaluated == summary.kl_total == 3
    assert summary.mean_kl == pytest.approx(0.02)
    assert summary.capability_passed is True
    assert summary.public_reported


def test_refusal_cap_rejects_before_full_set_and_forbids_kl(manifests) -> None:
    _, search, _ = manifests
    funnel = EvaluationFunnel()
    candidate = register(
        funnel,
        manifests,
        artifact=510,
        active_protocol=protocol(refusal_cap=0),
    )
    result = funnel.record_refusal(
        candidate,
        refusal(
            search.items[0].item_id,
            0,
            label=SemanticLabel.REFUSAL,
            lexical_hit=False,
        ),
    )
    assert result.next_stage is FunnelStage.REJECTED
    assert funnel.summary(candidate).refusal_evaluated == 1
    with pytest.raises(StageOrderError, match="next stage is rejected"):
        funnel.record_kl(candidate, kl(search.items[0].item_id, 0, 0.0))


def test_lexical_proxy_is_reported_but_semantic_verdict_controls_cap(manifests) -> None:
    _, search, _ = manifests
    funnel = EvaluationFunnel()
    candidate = register(
        funnel,
        manifests,
        artifact=520,
        active_protocol=protocol(refusal_cap=2),
    )
    observations = (
        refusal(
            search.items[0].item_id,
            0,
            label=SemanticLabel.COMPLIANCE,
            lexical_hit=True,
        ),
        refusal(
            search.items[1].item_id,
            1,
            label=SemanticLabel.PARTIAL_REFUSAL,
            lexical_hit=False,
            source=VerdictSource.TASK_ORACLE,
        ),
        refusal(
            search.items[2].item_id,
            2,
            label=SemanticLabel.DEFLECTION,
            lexical_hit=False,
            source=VerdictSource.HUMAN,
        ),
    )
    for observation in observations:
        funnel.record_refusal(candidate, observation)
    summary = funnel.summary(candidate)
    assert summary.lexical_proxy_hits == 1
    assert summary.semantic_refusals == 2
    assert summary.next_stage is FunnelStage.KL

    with pytest.raises(ValueError, match="unsupported"):
        RefusalObservation(
            item_id="x",
            response_sha256=sha(1),
            lexical_proxy_hit=False,
            semantic_label=SemanticLabel.COMPLIANCE,
            semantic_source="external-llm",  # type: ignore[arg-type]
            semantic_evidence_sha256=sha(2),
        )


def test_nonnegative_kl_lower_bound_prunes_exactly(manifests) -> None:
    _, search, _ = manifests
    funnel = EvaluationFunnel()
    candidate = register(
        funnel,
        manifests,
        artifact=530,
        active_protocol=protocol(refusal_cap=3, kl_cap=0.05),
    )
    complete_refusal(funnel, candidate, search)
    # 0.151 / 3 > 0.05, even if every unobserved row were exactly zero.
    result = funnel.record_kl(candidate, kl(search.items[0].item_id, 0, 0.151))
    assert result.next_stage is FunnelStage.REJECTED
    assert funnel.summary(candidate).kl_lower_bound > 0.05
    with pytest.raises(StageOrderError, match="next stage is rejected"):
        funnel.record_kl(candidate, kl(search.items[1].item_id, 1, 0.0))

    boundary = register(
        funnel,
        manifests,
        artifact=531,
        active_protocol=protocol(refusal_cap=3, kl_cap=0.05),
    )
    complete_refusal(funnel, boundary, search)
    funnel.record_kl(boundary, kl(search.items[0].item_id, 0, 0.15))
    assert funnel.next_stage(boundary) is FunnelStage.KL
    funnel.record_kl(boundary, kl(search.items[1].item_id, 1, 0.0))
    funnel.record_kl(boundary, kl(search.items[2].item_id, 2, 0.0))
    assert funnel.next_stage(boundary) is FunnelStage.CAPABILITY


@pytest.mark.parametrize("value", [-0.1, math.inf, -math.inf, math.nan])
def test_kl_rejects_invalid_values(value: float) -> None:
    with pytest.raises(ValueError, match="finite non-negative"):
        KLObservation("row", value, sha(1))


def test_failed_capability_blocks_public_report(manifests) -> None:
    _, search, _ = manifests
    funnel = EvaluationFunnel()
    candidate = register(
        funnel,
        manifests,
        artifact=540,
        active_protocol=protocol(refusal_cap=3, kl_cap=1.0),
    )
    complete_refusal(funnel, candidate, search)
    complete_kl(funnel, candidate, search, [0.1, 0.1, 0.1])
    assert funnel.record_capability(
        candidate,
        CapabilityObservation(False, sha(900)),
    ).next_stage is FunnelStage.REJECTED
    with pytest.raises(StageOrderError, match="next stage is rejected"):
        funnel.record_public_report(candidate, PublicReportObservation(sha(901)))


def test_immutable_cache_hits_conflicts_and_identity_isolation(manifests) -> None:
    _, search, _ = manifests
    funnel = EvaluationFunnel()
    active_protocol = protocol(refusal_cap=3)
    first_registration = funnel.register_candidate(
        artifact_sha256=sha(550),
        protocol=active_protocol,
        geometry_split=manifests[0],
        selection_split=search,
        public_test_split=manifests[2],
    )
    assert not first_registration.cache_hit
    replay = funnel.register_candidate(
        artifact_sha256=sha(550),
        protocol=active_protocol,
        geometry_split=manifests[0],
        selection_split=search,
        public_test_split=manifests[2],
    )
    assert replay.cache_hit
    assert replay.candidate == first_registration.candidate

    observation = refusal(search.items[0].item_id, 0)
    assert not funnel.record_refusal(replay.candidate, observation).cache_hit
    assert funnel.record_refusal(replay.candidate, observation).cache_hit
    conflict = RefusalObservation(
        item_id=observation.item_id,
        response_sha256=sha(999),
        lexical_proxy_hit=observation.lexical_proxy_hit,
        semantic_label=observation.semantic_label,
        semantic_source=observation.semantic_source,
        semantic_evidence_sha256=observation.semantic_evidence_sha256,
    )
    with pytest.raises(CacheConflictError, match="different evidence"):
        funnel.record_refusal(replay.candidate, conflict)

    other_artifact = register(
        funnel,
        manifests,
        artifact=551,
        active_protocol=active_protocol,
    )
    other_protocol = register(
        funnel,
        manifests,
        artifact=550,
        active_protocol=protocol(refusal_cap=3, salt=1),
    )
    assert other_artifact.sha256 != replay.candidate.sha256
    assert other_protocol.sha256 != replay.candidate.sha256
    assert funnel.cache_stats.hits == 2
    assert funnel.cache_stats.misses == 4
    assert funnel.cache_stats.hit_rate == pytest.approx(1 / 3)


def test_unknown_or_non_search_items_fail_closed(manifests) -> None:
    funnel = EvaluationFunnel()
    candidate = register(
        funnel,
        manifests,
        artifact=560,
        active_protocol=protocol(refusal_cap=3),
    )
    with pytest.raises(IdentityError, match="not in validation-search"):
        funnel.record_refusal(candidate, refusal("public-0", 0))
    with pytest.raises(IdentityError, match="unknown candidate"):
        funnel.next_stage(sha(999_999))


def test_frontier_is_nondominated_and_scoped_to_matched_protocol(manifests) -> None:
    _, search, _ = manifests
    funnel = EvaluationFunnel()
    active_protocol = protocol(refusal_cap=3, kl_cap=1.0)
    candidates = {}
    # A: (1, .04), B: (2, .02), C: (3, .05), D: (1, .06), E: (0, .08)
    configurations = {
        "a": (570, {0}, [0.04, 0.04, 0.04]),
        "b": (571, {0, 1}, [0.02, 0.02, 0.02]),
        "c": (572, {0, 1, 2}, [0.05, 0.05, 0.05]),
        "d": (573, {0}, [0.06, 0.06, 0.06]),
        "e": (574, set(), [0.08, 0.08, 0.08]),
    }
    for name, (artifact, refusal_indices, values) in configurations.items():
        candidate = register(
            funnel,
            manifests,
            artifact=artifact,
            active_protocol=active_protocol,
        )
        complete_refusal(
            funnel,
            candidate,
            search,
            refusal_indices=refusal_indices,
        )
        complete_kl(funnel, candidate, search, values)
        candidates[name] = candidate

    partial = register(
        funnel,
        manifests,
        artifact=575,
        active_protocol=active_protocol,
    )
    complete_refusal(funnel, partial, search)
    funnel.record_kl(partial, kl(search.items[0].item_id, 0, 0.01))

    frontier = funnel.frontier(candidates["a"])
    assert [point.candidate_sha256 for point in frontier] == [
        candidates["e"].sha256,
        candidates["a"].sha256,
        candidates["b"].sha256,
    ]
    assert [(point.semantic_refusals, point.mean_kl) for point in frontier] == [
        (0, pytest.approx(0.08)),
        (1, pytest.approx(0.04)),
        (2, pytest.approx(0.02)),
    ]
    assert partial.sha256 not in {point.candidate_sha256 for point in frontier}

    isolated = register(
        funnel,
        manifests,
        artifact=576,
        active_protocol=protocol(refusal_cap=3, kl_cap=1.0, salt=2),
    )
    complete_refusal(funnel, isolated, search)
    complete_kl(funnel, isolated, search, [0.0, 0.0, 0.0])
    assert isolated.sha256 not in {
        point.candidate_sha256 for point in funnel.frontier(candidates["a"])
    }


def test_public_report_cache_is_immutable_after_completion(manifests) -> None:
    _, search, _ = manifests
    funnel = EvaluationFunnel()
    candidate = register(
        funnel,
        manifests,
        artifact=580,
        active_protocol=protocol(refusal_cap=3, kl_cap=1.0),
    )
    complete_refusal(funnel, candidate, search)
    complete_kl(funnel, candidate, search, [0.0, 0.0, 0.0])
    capability = CapabilityObservation(True, sha(910))
    funnel.record_capability(candidate, capability)
    report = PublicReportObservation(sha(911))
    funnel.record_public_report(candidate, report)
    assert funnel.record_capability(candidate, capability).cache_hit
    assert funnel.record_public_report(candidate, report).cache_hit
    with pytest.raises(CacheConflictError):
        funnel.record_public_report(candidate, PublicReportObservation(sha(912)))
