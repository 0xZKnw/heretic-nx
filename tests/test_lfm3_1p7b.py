from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from experiments.lfm3_1p7b import (
    KL_CAP,
    choose_candidate,
    collect,
    fit_profiles,
    mean_kl,
    search,
    validate_layout,
)


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
