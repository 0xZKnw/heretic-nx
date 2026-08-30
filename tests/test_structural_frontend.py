from __future__ import annotations

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


class SplitAttention(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim // 2, bias=False)
        self.v_proj = nn.Linear(dim, dim // 2, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)


class SharedKVAttention(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)


class MLAAttention(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.q_a_proj = nn.Linear(dim, dim // 2, bias=False)
        self.q_b_proj = nn.Linear(dim // 2, dim, bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(dim, dim // 2, bias=False)
        self.kv_b_proj = nn.Linear(dim // 2, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)


class IOProjection(nn.Module):
    """Conv1D-compatible projection interface without importing Transformers."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.nx = input_dim
        self.nf = output_dim
        self.weight = nn.Parameter(torch.empty(input_dim, output_dim))


class FusedAttention(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.c_attn = nn.Linear(dim, 3 * dim, bias=False)
        self.c_proj = nn.Linear(dim, dim, bias=False)


class DenseFFN(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(dim, 2 * dim, bias=False)
        self.up_proj = nn.Linear(dim, 2 * dim, bias=False)
        self.down_proj = nn.Linear(2 * dim, dim, bias=False)


class GPT2FFN(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.c_fc = IOProjection(dim, 2 * dim)
        self.c_proj = IOProjection(2 * dim, dim)


class EOIDownProjection(nn.Module):
    def __init__(self, experts: int, input_dim: int, output_dim: int, *, dtype: torch.dtype) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(experts, output_dim, input_dim, dtype=dtype)
        )


class RoutedExperts(nn.Module):
    def __init__(self, dim: int, experts: int, *, dtype: torch.dtype) -> None:
        super().__init__()
        self.down_proj = EOIDownProjection(experts, 2 * dim, dim, dtype=dtype)


class DirectRoutedExperts(nn.Module):
    def __init__(self, dim: int, experts: int) -> None:
        super().__init__()
        self.w2 = nn.Parameter(torch.empty(experts, dim, 2 * dim))


class SharedExpert(nn.Module):
    def __init__(self, dim: int, *, dtype: torch.dtype) -> None:
        super().__init__()
        self.up_proj = nn.Linear(dim, 2 * dim, bias=False, dtype=dtype)
        self.down_proj = nn.Linear(2 * dim, dim, bias=False, dtype=dtype)


class MoEFFN(nn.Module):
    def __init__(self, dim: int, experts: int, *, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.gate = nn.Linear(dim, experts, bias=False, dtype=dtype)
        self.shared_expert = SharedExpert(dim, dtype=dtype)
        self.experts = RoutedExperts(dim, experts, dtype=dtype)


class ModuleListMoEFFN(nn.Module):
    def __init__(self, dim: int, experts: int) -> None:
        super().__init__()
        self.gate = nn.Linear(dim, experts, bias=False)
        self.experts = nn.ModuleList([DenseFFN(dim) for _ in range(experts)])


class DecoderLayer(nn.Module):
    def __init__(self, attention: nn.Module, ffn: nn.Module) -> None:
        super().__init__()
        self.attention_branch = attention
        self.feed_forward_branch = ffn
        self.input_layernorm = nn.LayerNorm(8)


class DirectFFNDecoderLayer(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.self_attn = SplitAttention(dim)
        self.fc1 = nn.Linear(dim, 2 * dim, bias=False)
        self.fc2 = nn.Linear(2 * dim, dim, bias=False)
        self.final_layer_norm = nn.LayerNorm(dim)


class SyntheticModel(nn.Module):
    def __init__(
        self,
        *,
        stack_path: str,
        attention: nn.Module | None = None,
        ffn: nn.Module | None = None,
        layers: int = 2,
    ) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=8)
        decoder_layers = nn.ModuleList(
            [
                DecoderLayer(
                    attention or SplitAttention(8),
                    ffn or DenseFFN(8),
                )
                for _ in range(layers)
            ]
        )
        owner: nn.Module = self
        parts = stack_path.split(".")
        for part in parts[:-1]:
            child = nn.Module()
            setattr(owner, part, child)
            owner = child
        setattr(owner, parts[-1], decoder_layers)


@pytest.mark.parametrize(
    "stack_path",
    (
        "model.layers",
        "backbone.layers",
        "transformer.blocks",
        "gpt_neox.layers",
    ),
)
def test_decoder_stack_discovery_is_wrapper_agnostic(stack_path: str) -> None:
    report = discover_structural_frontend(SyntheticModel(stack_path=stack_path))
    assert report.decoder_stack_path == stack_path
    assert report.layer_count == 2
    assert report.stream_dim == 8
    assert report.is_unambiguous
    assert len(report.activation_sites) == 4
    assert len(report.editable_targets) == 4
    assert {target.layout for target in report.weight_targets} == {WeightLayout.OI}
    assert len({site.id for site in report.activation_sites}) == 4
    assert len({target.id for target in report.weight_targets}) == 4


@pytest.mark.parametrize(
    ("attention", "variant"),
    (
        (SplitAttention(8), "split"),
        (FusedAttention(8), "fused"),
        (MLAAttention(8), "mla"),
    ),
)
def test_attention_is_recognized_by_residual_output_capability(
    attention: nn.Module,
    variant: str,
) -> None:
    report = discover_structural_frontend(
        SyntheticModel(stack_path="model.layers", attention=attention, layers=1)
    )
    site = next(site for site in report.activation_sites if site.role == "attention_output")
    target = report.targets_by_role("attention_output")[0]
    assert site.attention_variant == variant
    assert site.id == target.activation_site_id
    assert target.output_dim == 8


def test_unproven_io_like_module_fails_closed() -> None:
    inspected = inspect_structural_frontend(
        SyntheticModel(
            stack_path="transformer.blocks", ffn=GPT2FFN(8), layers=1
        )
    )
    assert any(
        item.code == "ffn_candidate_excluded"
        and "proven Tensor-output" in item.detail
        for item in inspected.exclusions
    )
    assert any(item.code == "ffn_output_not_found" for item in inspected.ambiguities)


def test_block_level_ffn_projections_are_discovered_without_architecture_branch() -> None:
    model = nn.Module()
    model.config = SimpleNamespace(hidden_size=8)
    model.decoder = nn.Module()
    model.decoder.layers = nn.ModuleList([DirectFFNDecoderLayer(8)])
    report = discover_structural_frontend(model)
    assert {target.role for target in report.weight_targets} == {
        "attention_output",
        "ffn_output",
    }
    ffn = report.targets_by_role("ffn_output")[0]
    assert ffn.module_path == "decoder.layers.0.fc2"
    assert ffn.layout is WeightLayout.OI


def test_moe_shared_and_routed_outputs_have_distinct_collision_free_ids() -> None:
    report = discover_structural_frontend(
        SyntheticModel(
            stack_path="backbone.layers",
            ffn=MoEFFN(8, 4),
            layers=1,
        )
    )
    shared = report.targets_by_role("shared_ffn_output")[0]
    routed = report.targets_by_role("routed_ffn_output")[0]
    assert shared.id != routed.id
    assert shared.layout is WeightLayout.OI
    assert shared.editable
    assert shared.activation_site_id is not None
    assert routed.layout is WeightLayout.EOI
    assert routed.shape == (4, 8, 16)
    assert routed.expert_count == 4
    assert not routed.editable
    assert routed.activation_site_id is None
    assert any(item.code == "routed_ffn_edit_deferred" for item in report.exclusions)
    assert any(item.code == "routed_ffn_bank_described" for item in report.inclusions)


def test_direct_eoi_parameter_is_described_without_inventing_activation_site() -> None:
    ffn = MoEFFN(8, 4)
    ffn.experts = DirectRoutedExperts(8, 4)
    report = discover_structural_frontend(
        SyntheticModel(stack_path="model.layers", ffn=ffn, layers=1)
    )
    routed = report.targets_by_role("routed_ffn_output")[0]
    assert routed.parameter_path.endswith("experts.w2")
    assert routed.module_path.endswith("experts")
    assert routed.layout is WeightLayout.EOI
    assert routed.expert_count == 4
    assert routed.activation_site_id is None
    assert not routed.editable


def test_module_list_experts_fail_closed_without_inventing_eoi_bank() -> None:
    inspected = inspect_structural_frontend(
        SyntheticModel(
            stack_path="model.layers",
            ffn=ModuleListMoEFFN(8, 3),
            layers=1,
        )
    )
    assert any(item.code == "ffn_output_not_unique" for item in inspected.ambiguities)
    assert not inspected.targets_by_role("routed_ffn_output")
    assert all(target.layout is not WeightLayout.EOI for target in inspected.weight_targets)
    with pytest.raises(StructuralDiscoveryError):
        discover_structural_frontend(
            SyntheticModel(
                stack_path="model.layers",
                ffn=ModuleListMoEFFN(8, 3),
                layers=1,
            )
        )


def test_structure_hash_covers_expert_count_layout_shape_and_dtype() -> None:
    base = discover_structural_frontend(
        SyntheticModel(stack_path="model.layers", ffn=MoEFFN(8, 3), layers=1)
    )
    same = discover_structural_frontend(
        SyntheticModel(stack_path="model.layers", ffn=MoEFFN(8, 3), layers=1)
    )
    more_experts = discover_structural_frontend(
        SyntheticModel(stack_path="model.layers", ffn=MoEFFN(8, 5), layers=1)
    )
    different_dtype = discover_structural_frontend(
        SyntheticModel(
            stack_path="model.layers",
            ffn=MoEFFN(8, 3, dtype=torch.float64),
            layers=1,
        )
    )
    dense_layout = discover_structural_frontend(
        SyntheticModel(stack_path="model.layers", ffn=DenseFFN(8), layers=1)
    )
    assert base.structure_hash == same.structure_hash
    assert len(
        {
            base.structure_hash,
            more_experts.structure_hash,
            different_dtype.structure_hash,
            dense_layout.structure_hash,
        }
    ) == 4


class AmbiguousAttention(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)


def test_ambiguous_projection_is_reported_and_discovery_fails_closed() -> None:
    model = SyntheticModel(
        stack_path="model.layers",
        attention=AmbiguousAttention(8),
        layers=1,
    )
    inspected = inspect_structural_frontend(model)
    assert not inspected.is_unambiguous
    assert any(item.code == "attention_output_not_unique" for item in inspected.ambiguities)
    assert inspected.inclusions
    with pytest.raises(StructuralDiscoveryError) as caught:
        discover_structural_frontend(model)
    assert caught.value.report == inspected


def test_multiple_decoder_collections_fail_closed_without_priority_heuristics() -> None:
    model = SyntheticModel(stack_path="model.layers", layers=1)
    model.backbone = nn.Module()
    model.backbone.layers = nn.ModuleList(
        [DecoderLayer(SplitAttention(8), DenseFFN(8))]
    )
    inspected = inspect_structural_frontend(model)
    assert inspected.decoder_stack_path is None
    assert inspected.ambiguities[0].code == "decoder_stack_not_unique"
    assert "backbone.layers" in inspected.ambiguities[0].detail
    assert "model.layers" in inspected.ambiguities[0].detail
    with pytest.raises(StructuralDiscoveryError):
        discover_structural_frontend(model)


def test_language_stack_is_selected_and_vision_stack_is_explicitly_excluded() -> None:
    model = SyntheticModel(stack_path="language_model.layers", layers=1)
    model.vision_tower = nn.Module()
    model.vision_tower.blocks = nn.ModuleList(
        [DecoderLayer(SplitAttention(8), DenseFFN(8))]
    )
    report = discover_structural_frontend(model)
    assert report.decoder_stack_path == "language_model.layers"
    exclusion = next(
        item
        for item in report.exclusions
        if item.code == "non_text_decoder_stack_excluded"
    )
    assert exclusion.path == "vision_tower.blocks"


def test_nested_text_config_supplies_stream_dimension() -> None:
    model = SyntheticModel(stack_path="backbone.layers", layers=1)
    model.config = SimpleNamespace(
        text_config=SimpleNamespace(hidden_size=8),
    )
    report = discover_structural_frontend(model)
    assert report.stream_dim == 8
    assert report.is_unambiguous


def test_tied_layer_targets_are_hash_stable_and_fail_closed() -> None:
    def tied_model() -> nn.Module:
        model = nn.Module()
        model.config = SimpleNamespace(hidden_size=8)
        model.model = nn.Module()
        layer = DecoderLayer(SplitAttention(8), DenseFFN(8))
        model.model.layers = nn.ModuleList([layer, layer])
        return model

    first = inspect_structural_frontend(tied_model())
    second = inspect_structural_frontend(tied_model())
    untied = inspect_structural_frontend(
        SyntheticModel(stack_path="model.layers", layers=2)
    )
    assert first.structure_hash == second.structure_hash
    assert first.structure_hash != untied.structure_hash
    aliases = [
        item for item in first.ambiguities if item.code == "weight_target_storage_shared"
    ]
    assert len(aliases) == 4
    assert all("model.layers.0" in item.detail for item in aliases)
    assert all("model.layers.1" in item.detail for item in aliases)
    with pytest.raises(StructuralDiscoveryError):
        discover_structural_frontend(tied_model())


def test_square_projection_without_declared_layout_fails_closed() -> None:
    class OpaqueProjection(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.empty(8, 8))

    model = SyntheticModel(stack_path="model.layers", layers=1)
    model.model.layers[0].attention_branch.o_proj = OpaqueProjection()
    inspected = inspect_structural_frontend(model)
    assert any(
        item.code == "attention_candidate_excluded"
        and "proven Tensor-output" in item.detail
        for item in inspected.exclusions
    )
    assert any(item.code == "attention_output_not_unique" for item in inspected.ambiguities)
    with pytest.raises(StructuralDiscoveryError):
        discover_structural_frontend(model)
