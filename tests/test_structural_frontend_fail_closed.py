"""Adversarial contracts for fail-closed structural model discovery.

These tests intentionally use small synthetic modules so that the safety
contracts run in the default CI environment without Transformers installed.
"""

from __future__ import annotations

from collections import OrderedDict
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from heretic_nx.model.structural_frontend import (
    StructuralDiscoveryError,
    WeightLayout,
    discover_structural_frontend,
    inspect_structural_frontend,
)


STREAM_DIM = 8


class SplitAttention(nn.Module):
    def __init__(self, dim: int = STREAM_DIM) -> None:
        super().__init__()
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)


class DenseFFN(nn.Module):
    def __init__(self, dim: int = STREAM_DIM) -> None:
        super().__init__()
        self.up_proj = nn.Linear(dim, 2 * dim, bias=False)
        self.down_proj = nn.Linear(2 * dim, dim, bias=False)


class DecoderLayer(nn.Module):
    def __init__(
        self,
        *,
        attention: nn.Module | None = None,
        ffn: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.self_attn = attention or SplitAttention()
        self.mlp = ffn or DenseFFN()
        self.input_layernorm = nn.LayerNorm(STREAM_DIM)


class DecoderModel(nn.Module):
    def __init__(self, layers: list[nn.Module]) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=STREAM_DIM)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(layers)


def _assert_discovery_fails_closed(model: nn.Module) -> None:
    report = inspect_structural_frontend(model)
    assert not report.is_unambiguous
    assert report.editable_targets == ()
    with pytest.raises(StructuralDiscoveryError):
        discover_structural_frontend(model)


def test_self_and_cross_attention_are_not_merged_into_one_role() -> None:
    class CrossAttentionLayer(DecoderLayer):
        def __init__(self) -> None:
            super().__init__()
            self.cross_attn = SplitAttention()

    model = DecoderModel([CrossAttentionLayer()])
    report = inspect_structural_frontend(model)

    assert not report.is_unambiguous
    ambiguity = next(
        item
        for item in report.ambiguities
        if (
            item.layer == 0
            and "attention" in item.code
            and "unique" in f"{item.code} {item.detail}"
        )
    )
    assert "model.layers.0.self_attn" in ambiguity.detail
    assert "model.layers.0.cross_attn" in ambiguity.detail
    assert report.editable_targets == ()
    with pytest.raises(StructuralDiscoveryError):
        discover_structural_frontend(model)


def test_custom_projection_does_not_prove_a_tensor_activation_boundary() -> None:
    class DictionaryProjection(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.in_features = STREAM_DIM
            self.out_features = STREAM_DIM
            self.weight = nn.Parameter(torch.empty(STREAM_DIM, STREAM_DIM))

        def forward(self, _inputs: torch.Tensor) -> dict[str, torch.Tensor]:
            return {"not_a_residual_tensor": self.weight}

    attention = SplitAttention()
    attention.o_proj = DictionaryProjection()
    model = DecoderModel([DecoderLayer(attention=attention)])
    report = inspect_structural_frontend(model)

    path = "model.layers.0.self_attn.o_proj"
    assert not any(target.module_path == path for target in report.weight_targets)
    assert any(item.path == path for item in report.exclusions)
    _assert_discovery_fails_closed(model)


def test_vision_only_stack_is_never_selected_as_a_text_decoder() -> None:
    model = nn.Module()
    model.config = SimpleNamespace(hidden_size=STREAM_DIM)
    model.vision_tower = nn.Module()
    model.vision_tower.blocks = nn.ModuleList([DecoderLayer()])

    report = inspect_structural_frontend(model)

    assert report.decoder_stack_path is None
    assert not report.is_unambiguous
    assert report.editable_targets == ()
    assert any(item.path == "vision_tower.blocks" for item in report.exclusions)


def test_multimodal_wrapper_selects_text_and_reports_vision_exclusion() -> None:
    model = nn.Module()
    model.config = SimpleNamespace(hidden_size=STREAM_DIM)
    model.language_model = nn.Module()
    model.language_model.layers = nn.ModuleList([DecoderLayer()])
    model.vision_tower = nn.Module()
    model.vision_tower.blocks = nn.ModuleList([DecoderLayer()])

    report = discover_structural_frontend(model)

    assert report.decoder_stack_path == "language_model.layers"
    assert report.is_unambiguous
    assert any(item.path == "vision_tower.blocks" for item in report.exclusions)
    assert all(
        target.parameter_path.startswith("language_model.layers.")
        for target in report.editable_targets
    )


def test_rank_three_attention_output_is_not_mislabelled_as_eoi_experts() -> None:
    attention = SplitAttention()
    attention.o_proj = nn.Conv1d(
        STREAM_DIM,
        STREAM_DIM,
        kernel_size=3,
        bias=False,
    )
    model = DecoderModel([DecoderLayer(attention=attention)])
    report = inspect_structural_frontend(model)

    path = "model.layers.0.self_attn.o_proj"
    assert not any(target.module_path == path for target in report.weight_targets)
    assert not any(
        target.role == "attention_output" and target.layout is WeightLayout.EOI
        for target in report.weight_targets
    )
    assert any(item.path == path for item in report.exclusions)
    _assert_discovery_fails_closed(model)


def test_distinct_parameters_with_shared_storage_fail_closed() -> None:
    model = DecoderModel([DecoderLayer(), DecoderLayer()])
    first = model.model.layers[0].self_attn.o_proj.weight
    alias = nn.Parameter(first.detach())
    model.model.layers[1].self_attn.o_proj.weight = alias

    assert first is not alias
    assert first.untyped_storage().data_ptr() == alias.untyped_storage().data_ptr()

    first_report = inspect_structural_frontend(model)
    second_report = inspect_structural_frontend(model)
    alias_records = [
        item
        for item in first_report.ambiguities
        if item.code == "weight_target_storage_shared"
    ]

    assert len(alias_records) == 2
    assert first_report.structure_hash == second_report.structure_hash
    assert first_report.structure_hash != inspect_structural_frontend(
        DecoderModel([DecoderLayer(), DecoderLayer()])
    ).structure_hash
    assert all("model.layers.0" in item.detail for item in alias_records)
    assert all("model.layers.1" in item.detail for item in alias_records)
    _assert_discovery_fails_closed(model)


def test_direct_shared_child_is_classified_as_shared_not_dense() -> None:
    class SharedLayer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.self_attn = SplitAttention()
            self.shared_mlp = DenseFFN()
            self.input_layernorm = nn.LayerNorm(STREAM_DIM)

    report = discover_structural_frontend(DecoderModel([SharedLayer()]))

    shared = report.targets_by_role("shared_ffn_output")
    assert len(shared) == 1
    assert shared[0].module_path.endswith("shared_mlp.down_proj")
    assert shared[0].editable
    assert not report.targets_by_role("ffn_output")


def test_direct_routed_child_is_deferred_instead_of_marked_dense() -> None:
    class RoutedLayer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.self_attn = SplitAttention()
            self.routed_experts = DenseFFN()
            self.input_layernorm = nn.LayerNorm(STREAM_DIM)

    report = discover_structural_frontend(DecoderModel([RoutedLayer()]))

    assert not report.targets_by_role("ffn_output")
    assert not report.targets_by_role("shared_ffn_output")
    assert not report.targets_by_role("routed_ffn_output")
    assert any(
        item.code == "per_expert_matrix_edit_deferred"
        and "routed_experts.down_proj" in item.path
        for item in report.exclusions
    )


def test_unclassified_residual_width_output_branch_is_ambiguous() -> None:
    class UnknownBranch(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.out_proj = nn.Linear(STREAM_DIM, STREAM_DIM, bias=False)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return self.out_proj(inputs)

    layer = DecoderLayer()
    layer.unknown_mixer = UnknownBranch()
    model = DecoderModel([layer])
    report = inspect_structural_frontend(model)

    assert any(
        item.path == "model.layers.0.unknown_mixer"
        for item in report.ambiguities
    )
    _assert_discovery_fails_closed(model)


def test_query_and_output_names_alone_do_not_prove_shared_kv_attention() -> None:
    class NotAttention(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q_proj = nn.Linear(STREAM_DIM, STREAM_DIM, bias=False)
            self.o_proj = nn.Linear(STREAM_DIM, STREAM_DIM, bias=False)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return self.o_proj(inputs)

    layer = DecoderLayer()
    del layer.self_attn
    layer.suspicious_branch = NotAttention()
    model = DecoderModel([layer])
    report = inspect_structural_frontend(model)

    assert not report.targets_by_role("attention_output")
    assert any(
        item.path == "model.layers.0.suspicious_branch"
        for item in report.ambiguities
    )
    _assert_discovery_fails_closed(model)


@pytest.mark.parametrize("unsupported", ("int8", "sparse"))
def test_unsupported_weight_storage_is_never_editable(unsupported: str) -> None:
    attention = SplitAttention()
    if unsupported == "int8":
        weight = torch.zeros(
            (STREAM_DIM, STREAM_DIM),
            dtype=torch.int8,
        )
    else:
        indices = torch.tensor([[0, 1], [0, 1]])
        values = torch.ones(2)
        weight = torch.sparse_coo_tensor(
            indices,
            values,
            (STREAM_DIM, STREAM_DIM),
            check_invariants=True,
        ).coalesce()
    attention.o_proj.weight = nn.Parameter(weight, requires_grad=False)
    model = DecoderModel([DecoderLayer(attention=attention)])

    report = inspect_structural_frontend(model)
    output_path = "model.layers.0.self_attn.o_proj.weight"

    assert not any(
        target.parameter_path == output_path and target.editable
        for target in report.weight_targets
    )
    assert not any(
        target.parameter_path == output_path
        for target in report.editable_targets
    )


def test_square_rank_three_routed_bank_has_no_inferred_eoi_layout() -> None:
    class SquareExpertBank(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w2 = nn.Parameter(torch.empty(2, STREAM_DIM, STREAM_DIM))

    class RoutedFFN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gate_up_proj = nn.Linear(STREAM_DIM, 2 * STREAM_DIM, bias=False)
            self.experts = SquareExpertBank()

    model = DecoderModel([DecoderLayer(ffn=RoutedFFN())])
    report = inspect_structural_frontend(model)

    assert not any(
        target.layout is WeightLayout.EOI
        and target.parameter_path.endswith("experts.w2")
        for target in report.weight_targets
    )
    assert any(
        "EOI" in item.detail and "EIO" in item.detail
        for item in (*report.exclusions, *report.ambiguities)
    )
    _assert_discovery_fails_closed(model)


def test_named_sequential_uses_registered_layer_paths() -> None:
    model = nn.Module()
    model.config = SimpleNamespace(hidden_size=STREAM_DIM)
    model.decoder = nn.Module()
    model.decoder.layers = nn.Sequential(
        OrderedDict(
            (
                ("first", DecoderLayer()),
                ("second", DecoderLayer()),
            )
        )
    )

    report = discover_structural_frontend(model)

    assert report.layer_count == 2
    assert any(
        "decoder.layers.first." in target.parameter_path
        for target in report.weight_targets
    )
    assert any(
        "decoder.layers.second." in target.parameter_path
        for target in report.weight_targets
    )


def test_target_storage_alias_with_non_target_parameter_fails_closed() -> None:
    model = DecoderModel([DecoderLayer()])
    attention = model.model.layers[0].self_attn
    attention.o_proj.weight = nn.Parameter(attention.q_proj.weight.detach())

    report = inspect_structural_frontend(model)
    alias = next(
        item
        for item in report.ambiguities
        if item.code == "weight_target_storage_shared"
    )
    assert "self_attn.o_proj.weight" in alias.detail
    assert "self_attn.q_proj.weight" in alias.detail
    _assert_discovery_fails_closed(model)


def test_same_attention_module_at_two_call_sites_fails_closed() -> None:
    layer = DecoderLayer()
    layer.cross_attn = layer.self_attn
    model = DecoderModel([layer])

    report = inspect_structural_frontend(model)
    ambiguity = next(
        item
        for item in report.ambiguities
        if item.code == "attention_output_not_unique"
    )
    assert "model.layers.0.self_attn" in ambiguity.detail
    assert "model.layers.0.cross_attn" in ambiguity.detail
    _assert_discovery_fails_closed(model)


def test_uninitialized_lazy_projection_returns_a_report_instead_of_crashing() -> None:
    attention = SplitAttention()
    attention.o_proj = nn.LazyLinear(STREAM_DIM, bias=False)
    model = DecoderModel([DecoderLayer(attention=attention)])

    report = inspect_structural_frontend(model)
    assert any("uninitialized" in item.detail for item in report.exclusions)
    _assert_discovery_fails_closed(model)


@pytest.mark.parametrize("mutation", ("call_override", "forward_hook"))
def test_runtime_output_mutation_invalidates_projection_contract(
    mutation: str,
) -> None:
    class CallOverrideLinear(nn.Linear):
        def __call__(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
            return {"not_a_residual": inputs}

    attention = SplitAttention()
    if mutation == "call_override":
        attention.o_proj = CallOverrideLinear(STREAM_DIM, STREAM_DIM, bias=False)
    else:
        attention.o_proj.register_forward_hook(
            lambda _module, _inputs, output: {"not_a_residual": output}
        )
    model = DecoderModel([DecoderLayer(attention=attention)])

    report = inspect_structural_frontend(model)
    assert any(
        item.path.endswith("self_attn.o_proj") for item in report.exclusions
    )
    _assert_discovery_fails_closed(model)


def test_projection_names_without_projection_contract_do_not_prove_mla() -> None:
    class FakeMLA(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q_proj = nn.ReLU()
            self.kv_a_proj = nn.ReLU()
            self.kv_b_proj = nn.ReLU()
            self.o_proj = nn.Linear(STREAM_DIM, STREAM_DIM, bias=False)

    model = DecoderModel([DecoderLayer(attention=FakeMLA())])
    report = inspect_structural_frontend(model)

    assert not report.targets_by_role("attention_output")
    _assert_discovery_fails_closed(model)


def test_ffn_output_requires_a_proven_expansion_path() -> None:
    class FakeFFN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.up_proj = nn.ReLU()
            self.down_proj = nn.Linear(STREAM_DIM, STREAM_DIM, bias=False)

    model = DecoderModel([DecoderLayer(ffn=FakeFFN())])
    report = inspect_structural_frontend(model)

    assert any(
        item.code == "ffn_structure_incoherent" for item in report.ambiguities
    )
    _assert_discovery_fails_closed(model)


def test_config_and_normalization_widths_must_agree() -> None:
    layer = DecoderLayer()
    layer.input_layernorm = nn.LayerNorm(2 * STREAM_DIM)
    model = DecoderModel([layer])

    report = inspect_structural_frontend(model)
    assert report.stream_dim is None
    assert any(
        item.code == "stream_dimension_not_unique"
        and "disagree" in item.detail
        for item in report.ambiguities
    )
    _assert_discovery_fails_closed(model)
    assert not any("decoder.layers.0." in target.parameter_path for target in report.weight_targets)
    assert not any("decoder.layers.1." in target.parameter_path for target in report.weight_targets)
