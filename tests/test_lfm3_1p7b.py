from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from experiments.lfm3_1p7b import (
    KL_CAP,
    bind_profile,
    choose_candidate,
    collect,
    compact_metrics,
    fit_profiles,
    mean_kl,
    sandwich_weight_operator,
    search,
    select_registry,
    validate_layout,
)
from heretic_nx.geometry.contrastive import ContrastiveAxis
from heretic_nx.hashing import sha256_json
from heretic_nx.model import SemanticSite, SemanticSiteRegistry


def test_delivered_metrics_hash_raw_responses_without_mutating_local_reports():
    report = {
        "arms": [{"responses": ["example answer"], "marker_count": 0}],
        "scale": 1.5,
    }
    before = deepcopy(report)
    assert compact_metrics(report) == {
        "arms": [
            {"responses_sha256": sha256_json(["example answer"]), "marker_count": 0}
        ],
        "scale": 1.5,
    }
    assert report == before


class Block(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.feed_forward = nn.Module()
        self.feed_forward.w2 = nn.Linear(2, 2048, bias=False)
        if kind == "conv":
            self.conv = nn.Module()
            self.conv.conv = nn.Conv1d(2, 2, 7)
            self.conv.in_proj = nn.Linear(2, 6)
            self.conv.out_proj = nn.Linear(2, 2048, bias=False)
        else:
            self.self_attn = nn.Module()
            for name in ("q_proj", "k_proj", "v_proj"):
                setattr(self.self_attn, name, nn.Linear(2, 2))
            self.self_attn.o_proj = nn.Linear(2, 2048, bias=False)

    def forward(self, values):
        return values + 1


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        types = ["conv"] * 13 + ["full_attention"] * 6 + ["sliding_attention"] * 5
        self.config = SimpleNamespace(
            model_type="lfm3",
            hidden_size=2048,
            layer_types=types,
            max_position_embeddings=4096,
        )
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([Block(kind) for kind in types])
        self.device = "cpu"

    def forward(self, input_ids, **kwargs):
        state = input_ids.float().unsqueeze(-1).expand(-1, -1, 2048)
        for layer in self.model.layers:
            state = layer(state)
        return state


class TokenBatch(dict):
    @property
    def input_ids(self):
        return self["input_ids"]

    def to(self, device):
        return self


class Tokenizer:
    def apply_chat_template(self, prompts, **kwargs):
        assert kwargs["enable_thinking"] is False
        return prompts

    def __call__(self, prompts, **kwargs):
        return TokenBatch(input_ids=torch.tensor([[0, 7]] * len(prompts)))


def test_all_three_lfm3_block_types_are_discovered_and_fail_closed():
    model = TinyModel()
    registry = validate_layout(model)
    assert len(registry.by_family("gqa")) == 11
    assert len(registry.by_family("liv")) == 13
    assert len(registry.by_family("ffn")) == 24
    del model.model.layers[-1].self_attn.o_proj
    with pytest.raises(ValueError, match="incomplete"):
        validate_layout(model)


def test_sparse_registry_keeps_only_late_mixers_without_mutating_the_full_layout():
    registry = validate_layout(TinyModel())
    selected = select_registry(registry, 8, "operator")
    assert len(selected.sites) == 16
    assert all(s.layer >= 8 and s.family in {"gqa", "liv"} for s in selected.sites)
    assert len(registry.by_family("ffn")) == 24
    assert len(select_registry(registry, 20, "ffn").sites) == 4


@pytest.mark.parametrize(
    "field,value",
    [("model_type", "lfm2"), ("hidden_size", 1024), ("layer_types", ["conv"] * 24)],
)
def test_unexpected_architecture_is_rejected(field, value):
    model = TinyModel()
    setattr(model.config, field, value)
    with pytest.raises(ValueError):
        validate_layout(model)


def test_collection_captures_each_raw_block_and_removes_hooks_on_failure():
    model = TinyModel()
    actual = collect(model, Tokenizer(), ["one", "two", "three"], 2)
    assert actual.shape == (3, 24, 2048)
    assert torch.equal(actual[:, :, 0], torch.arange(8, 32).expand(3, -1))
    assert all(not layer._forward_hooks for layer in model.model.layers)
    model.model.layers[1].forward = lambda _values: (_ for _ in ()).throw(
        RuntimeError("test")
    )
    with pytest.raises(RuntimeError, match="test"):
        collect(model, Tokenizer(), ["one"], 1)
    assert all(not layer._forward_hooks for layer in model.model.layers)


def test_search_never_overwrites_an_existing_checkpoint(tmp_path):
    sentinel = tmp_path / "keep.txt"
    sentinel.write_text("unchanged")
    with pytest.raises(FileExistsError):
        search(SimpleNamespace(output=tmp_path))
    assert sentinel.read_text() == "unchanged"


def test_zero_contrast_in_early_convolution_blocks_does_not_block_the_search():
    generator = torch.Generator().manual_seed(8)
    safe = torch.randn(18, 24, 8, generator=generator)
    target = safe + torch.randn(18, 24, 8, generator=generator) + 0.5
    safe[:, :4] = target[:, :4] = 0
    profiles = fit_profiles(safe, target)
    for axes in profiles.values():
        assert set(axes) == set(range(4, 24))
        assert all(
            float(axis.axis.norm()) == pytest.approx(1.0, abs=1e-5)
            for axis in axes.values()
        )


def test_candidate_selection_rejects_bad_kl_and_requires_improvement():
    rows = [
        {"marker_count": 0, "kl": value}
        for value in (-1, float("nan"), float("inf"), KL_CAP + 0.001)
    ]
    rows += [{"marker_count": 10, "kl": 0.0}]
    with pytest.raises(RuntimeError):
        choose_candidate(rows, 10)
    rows += [{"marker_count": 2, "kl": 0.01}, {"marker_count": 2, "kl": 0.001}]
    before = deepcopy(rows[-2:])
    assert choose_candidate(rows, 10) == rows[-1]
    assert rows[-2:] == before


def test_explicit_refusal_target_selects_the_least_drifting_successful_candidate():
    rows = [
        {"marker_count": 0, "kl": 0.2},
        {"marker_count": 1, "kl": 0.05},
        {"marker_count": 0, "kl": 0.1},
        {"marker_count": 0, "kl": 0.01, "empty_responses": 1},
    ]
    assert choose_candidate(rows, 22, kl_cap=0.5, max_refusals=0) is rows[2]
    assert choose_candidate(rows, 22, kl_cap=0.5, max_refusals=1) is rows[1]
    with pytest.raises(RuntimeError):
        choose_candidate(rows, 22, kl_cap=0.05, max_refusals=0)


def test_sandwich_operator_matches_the_residual_coordinate_projection():
    generator = torch.Generator().manual_seed(48)
    for _ in range(32):
        axis = torch.nn.functional.normalize(torch.randn(8, generator=generator), dim=0)
        gain = torch.randn(8, generator=generator).sign() * (
            torch.rand(8, generator=generator) + 0.1
        )
        norms = torch.rand(8, generator=generator) + 0.1
        hidden = torch.randn(4, 8, generator=generator)
        op = sandwich_weight_operator(axis, gain, norms)
        residual = hidden * (gain * norms)
        expected = residual - (residual @ axis[:, None]) * axis
        assert torch.allclose(
            op.apply(hidden) * (gain * norms), expected, atol=1e-6, rtol=1e-5
        )


@pytest.mark.parametrize("family", ["ffn", "gqa", "liv"])
def test_shared_axis_is_bound_in_the_correct_branch_coordinates(family):
    layer = SimpleNamespace(
        post_ffn_norm=nn.LayerNorm(4), post_operator_norm=nn.LayerNorm(4)
    )
    with torch.no_grad():
        layer.post_ffn_norm.weight.copy_(torch.tensor([0.5, 2, 1, 1]))
        layer.post_operator_norm.weight.fill_(10)
    model = SimpleNamespace(model=SimpleNamespace(layers=[None] * 4 + [layer]))
    site = SemanticSite("test", 4, family, "ffn_out", "test", "Linear", 4, 4, 4, "test")
    registry = SemanticSiteRegistry((site,), "test")
    local = ContrastiveAxis(torch.tensor([0.0, 0.0, 1.0, 0.0]), 1, 1, 0, 3)
    shared = ContrastiveAxis(torch.tensor([1.0, 1.0, 0.0, 0.0]) / 2**0.5, 1, 1, 0, 3)
    editors = bind_profile(
        model,
        registry,
        {4: local, 12: shared},
        {"test": torch.diag(torch.tensor([2.0, 3.0, 4.0, 5.0]))},
        shared_axis=12,
        sandwich_aware=True,
    )
    coordinates = (
        torch.tensor([1.0, 6.0, 4.0, 5.0])
        if family == "ffn"
        else torch.tensor([20.0, 30.0, 40.0, 50.0])
    )
    assert torch.allclose(editors[0].operator.a[:, 0] * coordinates, shared.axis)
    assert torch.allclose(editors[0].operator.b[:, 0] / coordinates, shared.axis)


@pytest.mark.parametrize(
    "bad", [torch.zeros(8), torch.full((8,), float("nan")), torch.ones(7)]
)
def test_sandwich_operator_rejects_singular_or_invalid_coordinates(bad):
    with pytest.raises(ValueError):
        sandwich_weight_operator(torch.ones(8), bad, torch.ones(8))


def test_kl_matches_hand_computed_distribution_and_rejects_invalid_inputs():
    base = torch.tensor([[0.5, 0.5]]).log()
    candidate = torch.tensor([[0.25, 0.75]]).log()
    assert mean_kl(base, base) == 0
    assert mean_kl(base, candidate) == pytest.approx(0.1438410362, abs=1e-6)
    for invalid in (candidate[:, :1], candidate[:0], candidate * float("nan")):
        with pytest.raises(ValueError):
            mean_kl(base, invalid)


def selection_contract(
    base: int, first: int, second: int, kl_first: float, kl_second: float
):
    """Symbolically explore the production selection gate, without tensor execution."""
    assert base >= 0 and first >= 0 and second >= 0
    rows = [
        {"marker_count": first, "kl": kl_first},
        {"marker_count": second, "kl": kl_second},
    ]
    eligible = [
        row for row in rows if row["marker_count"] < base and 0 <= row["kl"] <= KL_CAP
    ]
    try:
        result = choose_candidate(rows, base)
    except RuntimeError:
        assert not eligible
        return
    assert result in eligible
    assert result["marker_count"] < base
    assert 0 <= result["kl"] <= KL_CAP
    for other in eligible:
        assert (result["marker_count"], result["kl"]) <= (
            other["marker_count"],
            other["kl"],
        )


def target_selection_contract(
    base: int, first: int, second: int, kl_first: float, kl_second: float
):
    assert base >= 0 and first >= 0 and second >= 0
    rows = [
        {"marker_count": first, "kl": kl_first},
        {"marker_count": second, "kl": kl_second},
    ]
    eligible = [
        r
        for r in rows
        if r["marker_count"] == 0 and r["marker_count"] < base and 0 <= r["kl"] <= 0.5
    ]
    try:
        result = choose_candidate(rows, base, kl_cap=0.5, max_refusals=0)
    except RuntimeError:
        assert not eligible
        return
    assert result in eligible
    assert result["marker_count"] == 0
    for other in eligible:
        assert result["kl"] <= other["kl"]
