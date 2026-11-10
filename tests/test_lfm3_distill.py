from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file
from torch import nn

from experiments.lfm3_1p7b import publish_checkpoint
from experiments.lfm3_1p7b_distill import (
    apply_factors,
    cache_factors,
    collect_inputs,
    fit_detector,
)
from heretic_nx.optimize.ridge import conjugate_gradient


@pytest.mark.parametrize(
    "options",
    [
        {"steps": 0},
        {"tolerance": 0},
        {"ridge_fraction": -1},
        {"safe_lambda": -1},
        {"steps": True},
    ],
)
def test_shared_solver_rejects_invalid_iteration_and_regularization_controls(options):
    parameters = {"safe_lambda": 1.0, **options}
    with pytest.raises(ValueError):
        conjugate_gradient(
            torch.ones(4, 2), torch.ones(4, 2), torch.ones(2), **parameters
        )


@pytest.mark.parametrize("safe_weight", [1.0, 10.0, 100.0])
def test_detector_cg_matches_the_dense_regularized_solution(safe_weight):
    generator = torch.Generator().manual_seed(7)
    target = torch.randn(16, 4, generator=generator)
    safe = torch.randn(20, 4, generator=generator)
    exact = torch.randn(4, generator=generator)
    actual, iterations, residual = fit_detector(target, safe, exact, safe_weight)
    ridge = 1e-4 * (target.square().mean() + safe_weight * safe.square().mean())
    h = target.double().T @ target.double() / len(target)
    s = safe.double().T @ safe.double() / len(safe)
    expected = torch.linalg.solve(
        h + safe_weight * s + ridge * torch.eye(4), h @ exact.double()
    )
    assert torch.allclose(actual.double(), expected, atol=1e-5, rtol=1e-4)
    assert 1 <= iterations <= 40
    assert residual <= 1e-5


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_detector_rejects_invalid_penalties(bad):
    with pytest.raises(ValueError):
        fit_detector(torch.ones(4, 2), torch.ones(4, 2), torch.ones(2), bad)


def test_detector_rejects_nonfinite_and_misaligned_states():
    with pytest.raises(ValueError):
        fit_detector(torch.ones(4, 2), torch.ones(4, 3), torch.ones(2), 1)
    with pytest.raises(ValueError):
        fit_detector(
            torch.full((4, 2), float("nan")), torch.ones(4, 2), torch.ones(2), 1
        )


class Batch(dict):
    @property
    def input_ids(self):
        return self["input_ids"]

    def to(self, device):
        return self


class Tokenizer:
    eos_token_id = 9
    pad_token_id = 0

    def apply_chat_template(self, prompts, **kwargs):
        return prompts

    def __call__(self, prompts, **kwargs):
        return Batch(input_ids=torch.tensor([[0, 7]] * len(prompts)))


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(2, 2, bias=False)
        self.config = SimpleNamespace(max_position_embeddings=1024)
        self.device = "cpu"
        self.fail = False

    def forward(self, input_ids, **kwargs):
        if self.fail:
            raise RuntimeError("test failure")
        return self.layer(torch.ones(*input_ids.shape, 2))

    def generate(self, input_ids, **kwargs):
        for step in range(8):
            ids = input_ids.clone()
            if step >= 2:
                ids[0, -1] = 9
            self(input_ids=ids)
        return input_ids


def test_projection_capture_filters_finished_rows_and_always_removes_hooks():
    model = Model()
    sites = [SimpleNamespace(id="one", module_path="layer")]
    safe = collect_inputs(model, Tokenizer(), ["a", "b"], sites)
    assert torch.equal(safe["one"], torch.ones(2, 2, dtype=torch.bfloat16))
    target = collect_inputs(model, Tokenizer(), ["a", "b"], sites, trajectory=True)
    assert target["one"].shape == (3, 2)
    assert not model._forward_pre_hooks and not model.layer._forward_pre_hooks
    model.fail = True
    with pytest.raises(RuntimeError, match="test failure"):
        collect_inputs(model, Tokenizer(), ["a"], sites)
    assert not model._forward_pre_hooks and not model.layer._forward_pre_hooks


def test_additive_candidates_restore_originals_instead_of_accumulating():
    model = Model()
    site = SimpleNamespace(id="one", module_path="layer")
    originals = {"one": torch.tensor([[2.0, 0.0], [0.0, 3.0]])}
    factors = {
        "lambda1.one.axis": torch.tensor([1.0, 0.0]),
        "lambda1.one.detector": torch.tensor([0.5, -0.5]),
    }
    apply_factors(model, [site], originals, factors, 1, 1)
    apply_factors(model, [site], originals, factors, 1, 2)
    assert torch.equal(model.layer.weight, torch.tensor([[3.0, -1.0], [0.0, 3.0]]))
    with pytest.raises(ValueError):
        apply_factors(model, [site], originals, factors, 1, float("nan"))


def test_shared_publisher_preserves_an_existing_output(tmp_path):
    with pytest.raises(FileExistsError):
        publish_checkpoint(SimpleNamespace(output=tmp_path), None, None, {})


def test_factor_cache_roundtrips_axes_shared_across_penalty_trials(tmp_path):
    axis = torch.tensor([1.0, 2.0])
    payload = {
        "lambda1.axis": axis,
        "lambda10.axis": axis,
        "detector": torch.tensor([3.0, 4.0]),
    }
    with pytest.raises(RuntimeError, match="share memory"):
        save_file(payload, tmp_path / "bad.safetensors")
    cache_factors(payload, tmp_path / "factors.safetensors", "test")
    actual = load_file(tmp_path / "factors.safetensors")
    assert set(actual) == set(payload)
    assert all(torch.equal(actual[key], value) for key, value in payload.items())
