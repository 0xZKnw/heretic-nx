from __future__ import annotations

import torch
from torch import nn

from heretic_nx.edits.residual_stream import (
    apply_residual_stream_weight_edits,
    build_residual_stream_weight_editors,
    snapshot_residual_stream_weights,
)
from heretic_nx.geometry.contrastive import fit_contrastive_axis
from heretic_nx.model.semantic_sites import assert_lfm25_layout, discover_semantic_sites
from heretic_nx.runtime.golden import GoldenTensor, compare_golden, tensor_sha256


class TinyMLP(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, 2 * dim, bias=False)
        self.w3 = nn.Linear(dim, 2 * dim, bias=False)
        self.w2 = nn.Linear(2 * dim, dim, bias=False)


class TinyAttention(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim // 2, bias=False)
        self.v_proj = nn.Linear(dim, dim // 2, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)


class TinyConv(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, groups=dim)
        self.in_proj = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)


class TinyLayer(nn.Module):
    def __init__(self, dim: int, attention: bool) -> None:
        super().__init__()
        if attention:
            self.self_attn = TinyAttention(dim)
        else:
            self.conv = TinyConv(dim)
        self.feed_forward = TinyMLP(dim)


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = type("Config", (), {"hidden_size": 8})()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([TinyLayer(8, False), TinyLayer(8, True)])


class LlamaStyleMLP(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(dim, 2 * dim, bias=False)
        self.up_proj = nn.Linear(dim, 2 * dim, bias=False)
        self.down_proj = nn.Linear(2 * dim, dim, bias=False)


class LlamaStyleAttention(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim // 2, bias=False)
        self.v_proj = nn.Linear(dim, dim // 2, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)


class LlamaStyleLayer(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.self_attn = LlamaStyleAttention(dim)
        self.mlp = LlamaStyleMLP(dim)


class LlamaStyleModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = type("Config", (), {"hidden_size": 8})()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([LlamaStyleLayer(8), LlamaStyleLayer(8)])


def test_semantic_registry_uses_structure_and_is_deterministic() -> None:
    model = TinyModel()
    first = discover_semantic_sites(model)
    second = discover_semantic_sites(model)
    assert first == second
    assert len(first.by_family("liv")) == 1
    assert len(first.by_family("gqa")) == 1
    assert len(first.by_family("ffn")) == 2
    assert len(first.by_family("block")) == 2


def test_lfm_layout_assertion_fails_closed() -> None:
    registry = discover_semantic_sites(TinyModel())
    try:
        assert_lfm25_layout(registry)
    except RuntimeError as error:
        assert "unexpected LFM2.5" in str(error)
    else:
        raise AssertionError("tiny layout must not pass the LFM2.5 guard")


def test_semantic_registry_supports_llama_and_gemma_style_names() -> None:
    registry = discover_semantic_sites(LlamaStyleModel())
    assert len(registry.by_family("gqa")) == 2
    assert len(registry.by_family("ffn")) == 2
    assert registry.by_family("gqa")[0].module_path.endswith("self_attn.o_proj")
    assert registry.by_family("ffn")[0].module_path.endswith("mlp.down_proj")


def test_residual_stream_portfolio_is_architecture_neutral() -> None:
    model = LlamaStyleModel()
    registry = discover_semantic_sites(model)
    generator = torch.Generator().manual_seed(131)
    axes = []
    for layer in range(2):
        safe = torch.randn(30, 8, generator=generator)
        target = safe + torch.eye(8)[layer]
        axes.append(
            fit_contrastive_axis(
                safe, target, folds=3, remove_safe_mean=False
            )
        )
    editors = build_residual_stream_weight_editors(registry, axes)
    assert len(editors) == 4
    assert {editor.site.family for editor in editors} == {"gqa", "ffn"}
    originals = snapshot_residual_stream_weights(model, editors)
    strengths = {editor.site_id: 0.4 for editor in editors}
    apply_residual_stream_weight_edits(model, editors, originals, strengths)
    for editor in editors:
        edited = model.get_submodule(editor.module_path).weight.detach()
        original = originals[editor.site_id]
        torch.testing.assert_close(
            edited.float().norm(dim=1),
            original.float().norm(dim=1),
            atol=1e-5,
            rtol=1e-5,
        )
        assert not torch.equal(edited.cpu(), original)


def test_golden_fingerprints_and_tolerances() -> None:
    value = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    assert tensor_sha256(value) == tensor_sha256(value.clone())
    golden = GoldenTensor.capture("logits", value)
    assert golden.shape == (2, 3)
    passed = compare_golden(value + 1e-5, value, atol=2e-5, rtol=0)
    failed = compare_golden(value + 1e-3, value, atol=2e-5, rtol=0)
    assert passed.within_tolerance
    assert not failed.within_tolerance
