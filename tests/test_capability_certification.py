from __future__ import annotations

import torch

from heretic_nx.eval.capability import (
    certify_artifact_set,
    certify_capability_preservation,
    paired_bootstrap_interval,
    sequence_drift_between_models,
    teacher_forced_sequence_kl,
)


def test_teacher_forced_sequence_kl_uses_all_selected_tokens() -> None:
    baseline = torch.tensor(
        [
            [[4.0, 0.0, -1.0], [0.0, 4.0, -1.0], [1.0, 1.0, 1.0]],
            [[0.0, 0.0, 4.0], [4.0, 0.0, 0.0], [0.0, 4.0, 0.0]],
        ]
    )
    candidate = baseline.clone()
    candidate[0, 1] = torch.tensor([4.0, 0.0, -1.0])
    mask = torch.tensor([[True, True, False], [True, False, False]])
    drift = teacher_forced_sequence_kl(baseline, candidate, mask, top_k=2)
    assert drift.token_count == 3
    assert drift.mean_token_kl > 0
    assert drift.maximum_sequence_kl > drift.mean_token_kl
    assert 0 < drift.mean_topk_mass_coverage <= 1


def test_sequence_drift_handles_left_and_right_padding() -> None:
    class Batch(dict):
        def to(self, device):
            return Batch({key: value.to(device) for key, value in self.items()})

    class Tokenizer:
        def __call__(self, prompts, **_kwargs):
            assert tuple(prompts) == ("left", "right")
            return Batch(
                input_ids=torch.tensor([[0, 1, 2], [3, 4, 0]]),
                attention_mask=torch.tensor([[0, 1, 1], [1, 1, 0]]),
            )

    class TinyLM(torch.nn.Module):
        def __init__(self, shift: float) -> None:
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)
            self.shift = shift

        def forward(self, input_ids, attention_mask, use_cache=False):
            del attention_mask, use_cache
            logits = torch.nn.functional.one_hot(input_ids, num_classes=5).float() * 3
            logits[..., 0] += self.shift
            return type("Output", (), {"logits": logits})()

    drift = sequence_drift_between_models(
        TinyLM(0.0),
        TinyLM(0.2),
        Tokenizer(),
        ("left", "right"),
        batch_size=2,
    )
    assert drift.token_count == 2
    assert drift.mean_token_kl > 0


def test_capability_requires_simultaneous_noninferiority_and_sequence_kl() -> None:
    baseline = {
        "reasoning": torch.linspace(0.5, 1.0, 80),
        "coding": torch.linspace(0.4, 0.9, 80),
    }
    preserved = {name: values - 0.005 for name, values in baseline.items()}
    logits = torch.randn(2, 4, 7, generator=torch.Generator().manual_seed(173))
    drift = teacher_forced_sequence_kl(logits, logits + 0.001, torch.ones(2, 4, dtype=torch.bool))
    certificate = certify_capability_preservation(
        baseline,
        preserved,
        {"reasoning": 0.02, "coding": 0.02},
        drift,
        sequence_kl_maximum=0.01,
        resamples=1000,
        seed=173,
    )
    assert certificate.passed
    degraded = {**preserved, "coding": baseline["coding"] - 0.08}
    rejected = certify_capability_preservation(
        baseline,
        degraded,
        {"reasoning": 0.02, "coding": 0.02},
        drift,
        sequence_kl_maximum=0.01,
        resamples=1000,
        seed=173,
    )
    assert not rejected.passed
    assert rejected.blocking_reasons == ("slice:coding",)


def test_paired_bootstrap_detects_equivalence_only_inside_both_margins() -> None:
    baseline = torch.arange(100, dtype=torch.float64) / 100
    same = paired_bootstrap_interval(
        baseline,
        baseline + 0.005,
        margin=0.01,
        resamples=1000,
        seed=179,
    )
    assert same.noninferiority_passed and same.equivalence_passed
    improved = paired_bootstrap_interval(
        baseline,
        baseline + 0.05,
        margin=0.01,
        resamples=1000,
        seed=179,
    )
    assert improved.noninferiority_passed and not improved.equivalence_passed


def test_every_distributed_quantization_requires_its_own_certificate() -> None:
    baseline = {"reasoning": torch.linspace(0.5, 1.0, 40)}
    logits = torch.randn(1, 3, 5, generator=torch.Generator().manual_seed(181))
    drift = teacher_forced_sequence_kl(logits, logits, torch.ones(1, 3, dtype=torch.bool))
    certificate = certify_capability_preservation(
        baseline,
        baseline,
        {"reasoning": 0.01},
        drift,
        sequence_kl_maximum=0.01,
        resamples=500,
    )
    missing_quant = certify_artifact_set(
        {"bf16": certificate},
        required_artifacts=("bf16", "q4_k_m"),
    )
    assert not missing_quant.passed
    assert missing_quant.blocking_artifacts == ("q4_k_m",)
    complete = certify_artifact_set(
        {"bf16": certificate, "q4_k_m": certificate},
        required_artifacts=("bf16", "q4_k_m"),
    )
    assert complete.passed
