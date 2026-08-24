from __future__ import annotations

import torch

from heretic_nx.edits.affine import affine_operator_from_leace
from heretic_nx.edits.activation_op import ActivationOperator, metric_projector_operator
from heretic_nx.edits.matrix_opt import fit_low_rank_matrix_operator
from heretic_nx.edits.norm_preserving import norm_preserving_weight_edit
from heretic_nx.edits.nx_ir2 import (
    ActivationEditIR,
    NXIR2,
    RoutePolicyIR,
    SemanticSiteRef,
    ThinkClosePolicyIR,
)
from heretic_nx.edits.projector import input_projector_factors
from heretic_nx.edits.spectral import fit_signed_spectral_operator
from heretic_nx.edits.sparse import atomic_unit_scores, select_atomic_units
from heretic_nx.geometry.metric import LowRankMetric
from heretic_nx.geometry.leace import fit_leace
from heretic_nx.geometry.principal_angles import orthonormal_basis
from heretic_nx.runtime.temporal import BoundedPIDController, TemporalGate


def test_activation_metric_operator_matches_dense_formula() -> None:
    generator = torch.Generator().manual_seed(127)
    metric = LowRankMetric.from_factors(
        9,
        covariance_factor=torch.randn(9, 3, generator=generator),
    )
    q = torch.randn(9, 2, generator=generator)
    operator = metric_projector_operator(q, metric, beta=0.4)
    hidden = torch.randn(7, 9, generator=generator)
    gram = q.T @ metric.apply(q)
    dense_column = q @ torch.linalg.solve(gram + 1e-6 * torch.eye(2), q.T @ metric.dense())
    expected = hidden - 0.4 * hidden @ dense_column.T
    torch.testing.assert_close(operator.apply(hidden), expected, atol=2e-5, rtol=2e-5)


def test_activation_operator_spectral_bound_matches_dense_map() -> None:
    generator = torch.Generator().manual_seed(193)
    operator = metric_projector_operator(
        torch.randn(9, 2, generator=generator),
        LowRankMetric.from_factors(
            9,
            covariance_factor=torch.randn(9, 4, generator=generator),
        ),
        beta=0.7,
    )
    dense_norm = float(torch.linalg.matrix_norm(operator.a @ operator.b.T, ord=2))
    assert abs(operator.spectral_norm() - dense_norm) < 1e-5
    bounded = operator.bounded(0.8)
    assert bounded.spectral_norm() <= 0.80001
    assert bounded.beta == operator.beta


def test_norm_preserving_weight_edit_restores_every_output_row() -> None:
    generator = torch.Generator().manual_seed(197)
    weight = torch.randn(13, 17, generator=generator)
    q = orthonormal_basis(torch.randn(13, 2, generator=generator))
    operator = metric_projector_operator(
        q,
        LowRankMetric.from_factors(13, regularization=1.0),
        beta=1.0,
    )
    edited = norm_preserving_weight_edit(weight, operator, strength=1.6)
    torch.testing.assert_close(
        torch.linalg.vector_norm(edited, dim=1),
        torch.linalg.vector_norm(weight, dim=1),
        atol=2e-6,
        rtol=2e-6,
    )
    assert float(torch.linalg.vector_norm(edited - weight)) > 0


def test_norm_preserving_weight_edit_fails_on_collapsed_nonzero_row() -> None:
    weight = torch.eye(3)
    direction = torch.tensor([[1.0], [0.0], [0.0]])
    operator = ActivationOperator(
        a=direction,
        b=direction,
        beta=1.0,
    )
    try:
        norm_preserving_weight_edit(weight, operator, strength=1.0)
    except RuntimeError as error:
        assert "collapsed" in str(error)
    else:
        raise AssertionError("a collapsed non-zero row must fail closed")


def test_input_projector_factorization() -> None:
    generator = torch.Generator().manual_seed(131)
    weight = torch.randn(11, 8, generator=generator)
    q = orthonormal_basis(torch.randn(8, 3, generator=generator))
    factors = input_projector_factors(weight, q, 0.25)
    expected = weight @ (torch.eye(8) - 0.25 * q @ q.T)
    torch.testing.assert_close(factors.apply_weight(weight), expected)


def test_atomic_unit_selection_is_deterministic() -> None:
    scores = atomic_unit_scores(
        torch.tensor([1.0, 2.0, 2.0, 0.5]),
        torch.ones(4),
        torch.ones(4),
    )
    torch.testing.assert_close(select_atomic_units(scores, 2), torch.tensor([1, 2]))


def test_temporal_gate_budget_and_risk_shutdown() -> None:
    gate = TemporalGate(budget_tokens=3, maximum_window_tokens=2)
    assert gate.step(task_score=0.8, risk_score=0.1) == 1
    assert gate.step(task_score=0.0, risk_score=0.1, checkpoint=False) == 1
    assert gate.step(task_score=0.0, risk_score=0.1, checkpoint=False) == 0
    assert gate.step(task_score=0.8, risk_score=0.1) == 1
    assert gate.step(task_score=0.8, risk_score=0.9) == 0
    assert gate.shut_down


def test_pid_is_rate_limited_and_shuts_down() -> None:
    pid = BoundedPIDController(1.0, 0.1, 0.0, beta_max=0.8, rate_limit=0.2, integral_limit=2)
    assert pid.step(1.0) == 0.2
    assert pid.step(1.0) == 0.4
    assert pid.step(1.0, risk_shutdown=True) == 0.0


def test_nxir2_round_trip(tmp_path) -> None:
    site = SemanticSiteRef(
        id="L02:attention_out",
        layer=2,
        family="gqa",
        kind="attention_out",
        module_path="model.layers.2.self_attn.out_proj",
        module_type="Linear",
        stream_dim=8,
        structure_hash="1" * 64,
    )
    document = NXIR2(
        base_model_id="example/model",
        base_model_revision="abc",
        base_model_sha256="2" * 64,
        tokenizer_sha256="3" * 64,
        chat_template_sha256="4" * 64,
        frozen_manifest_sha256="5" * 64,
        semantic_registry_sha256="6" * 64,
        tensor_artifact_sha256="7" * 64,
        edits=(
            ActivationEditIR(
                site=site,
                family="metric_projector",
                rank=2,
                beta=0.3,
                a_key="site.a",
                b_key="site.b",
                protected_subspace_sha256="8" * 64,
            ),
        ),
    )
    path = tmp_path / "nx-ir2.json"
    document.write(path)
    assert NXIR2.read(path) == document
    assert NXIR2.read(path).content_id == document.content_id


def test_nxir2_supports_fail_closed_temporal_only_sidecar(tmp_path) -> None:
    route = RoutePolicyIR(
        id="safe-task-route",
        risk_probe_key="router.risk",
        task_probe_key="router.task",
        risk_threshold=0.4,
        task_threshold=0.8,
        calibration_sha256="9" * 64,
    )
    document = NXIR2(
        base_model_id="example/thinking-model",
        base_model_revision="abc",
        base_model_sha256="2" * 64,
        tokenizer_sha256="3" * 64,
        chat_template_sha256="4" * 64,
        frozen_manifest_sha256="5" * 64,
        semantic_registry_sha256="6" * 64,
        tensor_artifact_sha256="7" * 64,
        routes=(route,),
        generation_controls=(
            ThinkClosePolicyIR(
                id="close-think-96",
                route_id=route.id,
                open_token_id=64400,
                close_token_id=64401,
                budget_tokens=96,
                grace_tokens=8,
                close_logit_boost=12.0,
            ),
        ),
    )
    path = tmp_path / "temporal-only.nx-ir2.json"
    document.write(path)
    assert NXIR2.read(path) == document


def test_leace_affine_operator_matches_closed_form() -> None:
    generator = torch.Generator().manual_seed(163)
    labels = torch.randint(0, 2, (200,), generator=generator)
    concepts = torch.nn.functional.one_hot(labels, 2).float()
    values = concepts @ torch.randn(2, 7, generator=generator) + 0.2 * torch.randn(
        200, 7, generator=generator
    )
    eraser = fit_leace(values, concepts)
    operator = affine_operator_from_leace(eraser)
    torch.testing.assert_close(operator.apply(values), eraser.apply(values), atol=2e-5, rtol=2e-5)


def test_signed_spectral_operator_prefers_target_specific_direction() -> None:
    target = torch.zeros(6, 2)
    protected = torch.zeros(6, 2)
    target[4, 0] = 5.0
    protected[1, 0] = 5.0
    edit = fit_signed_spectral_operator(target, protected, rank=1, beta=0.5)
    assert int(edit.basis[:, 0].abs().argmax()) == 4
    hidden = torch.eye(6)
    edited = edit.operator.apply(hidden)
    assert float(edited[4, 4]) < 1.0
    torch.testing.assert_close(edited[1, 1], torch.tensor(1.0))


def test_low_rank_matrix_optimizer_reduces_target_separation_with_bounded_drift() -> None:
    generator = torch.Generator().manual_seed(167)
    protected = 0.2 * torch.randn(80, 8, generator=generator)
    target = 0.2 * torch.randn(80, 8, generator=generator)
    target[:, 0] += 2.0
    result = fit_low_rank_matrix_operator(
        target,
        protected,
        rank=2,
        beta=0.8,
        steps=80,
        seed=167,
    )
    assert result.final_loss < result.initial_loss
    assert result.target_separation_ratio < 0.5
    assert torch.linalg.matrix_norm(
        result.operator.a @ result.operator.b.T, ord=2
    ) <= 1.0001
