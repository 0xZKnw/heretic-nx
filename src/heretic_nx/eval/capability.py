"""Paired capability preservation and teacher-forced sequence drift metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True)
class PairedInterval:
    count: int
    mean_difference: float
    lower: float
    upper: float
    margin: float
    noninferiority_passed: bool
    equivalence_passed: bool


@dataclass(frozen=True)
class SequenceDrift:
    token_count: int
    mean_token_kl: float
    maximum_sequence_kl: float
    mean_topk_mass_coverage: float


@dataclass(frozen=True)
class CapabilityCertificate:
    passed: bool
    simultaneous_alpha: float
    slices: dict[str, PairedInterval]
    sequence_drift: SequenceDrift
    sequence_kl_maximum: float
    blocking_reasons: tuple[str, ...]


@dataclass(frozen=True)
class ArtifactCapabilitySet:
    passed: bool
    required_artifacts: tuple[str, ...]
    certificates: dict[str, CapabilityCertificate]
    blocking_artifacts: tuple[str, ...]


def paired_bootstrap_interval(
    baseline: Sequence[float] | Tensor,
    candidate: Sequence[float] | Tensor,
    *,
    margin: float,
    alpha: float = 0.05,
    resamples: int = 10_000,
    seed: int = 0,
) -> PairedInterval:
    """Paired percentile interval for candidate-minus-baseline mean score."""

    base = np.asarray(torch.as_tensor(baseline, dtype=torch.float64).cpu())
    edited = np.asarray(torch.as_tensor(candidate, dtype=torch.float64).cpu())
    if base.ndim != 1 or edited.shape != base.shape or base.size < 2:
        raise ValueError("paired scores must be aligned vectors with at least two rows")
    if not np.isfinite(base).all() or not np.isfinite(edited).all():
        raise ValueError("paired scores must be finite")
    if margin < 0 or not 0 < alpha < 1 or resamples < 100:
        raise ValueError("margin, alpha, or resamples is invalid")
    differences = edited - base
    generator = np.random.default_rng(seed)
    means = np.empty(resamples, dtype=np.float64)
    for start in range(0, resamples, 1024):
        stop = min(start + 1024, resamples)
        indices = generator.integers(
            0, differences.size, size=(stop - start, differences.size)
        )
        means[start:stop] = differences[indices].mean(axis=1)
    lower, upper = np.quantile(means, (alpha / 2, 1 - alpha / 2))
    return PairedInterval(
        count=int(differences.size),
        mean_difference=float(differences.mean()),
        lower=float(lower),
        upper=float(upper),
        margin=float(margin),
        noninferiority_passed=bool(lower >= -margin),
        equivalence_passed=bool(lower >= -margin and upper <= margin),
    )


def teacher_forced_sequence_kl(
    baseline_logits: Tensor,
    candidate_logits: Tensor,
    token_mask: Tensor,
    *,
    top_k: int = 128,
) -> SequenceDrift:
    """Compute full-vocabulary KL on every selected teacher-forced token."""

    if baseline_logits.ndim != 3 or candidate_logits.shape != baseline_logits.shape:
        raise ValueError("baseline and candidate logits must be aligned [batch, token, vocab]")
    if token_mask.shape != baseline_logits.shape[:2]:
        raise ValueError("token_mask must align with batch and token dimensions")
    if baseline_logits.device != candidate_logits.device or baseline_logits.device != token_mask.device:
        raise ValueError("sequence KL tensors must be on the same device")
    if not torch.isfinite(baseline_logits).all() or not torch.isfinite(candidate_logits).all():
        raise ValueError("sequence KL logits must be finite")
    if top_k < 1:
        raise ValueError("top_k must be positive")
    top_k = min(top_k, baseline_logits.shape[-1])
    mask = token_mask.bool()
    if not mask.any():
        raise ValueError("token_mask must select at least one token")
    base_log_probs = torch.log_softmax(baseline_logits.float(), dim=-1)
    candidate_log_probs = torch.log_softmax(candidate_logits.float(), dim=-1)
    base_probs = base_log_probs.exp()
    per_token = torch.sum(base_probs * (base_log_probs - candidate_log_probs), dim=-1)
    selected = per_token[mask]
    counts = mask.sum(dim=1)
    valid_sequences = counts > 0
    sequence_means = (per_token * mask).sum(dim=1)[valid_sequences] / counts[valid_sequences]
    top_indices = torch.topk(base_log_probs, k=top_k, dim=-1).indices
    top_mass = torch.gather(base_probs, -1, top_indices).sum(dim=-1)[mask]
    return SequenceDrift(
        token_count=int(mask.sum().item()),
        mean_token_kl=float(selected.mean().item()),
        maximum_sequence_kl=float(sequence_means.max().item()),
        mean_topk_mass_coverage=float(top_mass.mean().item()),
    )


@torch.inference_mode()
def sequence_drift_between_models(
    baseline,
    candidate,
    tokenizer,
    rendered: Sequence[str],
    *,
    batch_size: int = 1,
    max_length: int = 512,
) -> SequenceDrift:
    """Aggregate exact teacher-forced KL for two compatible causal LMs.

    The last non-padding position is excluded because its next token is not
    present in the teacher-forced input. This remains correct for both left
    and right padding.
    """

    prompts = tuple(rendered)
    if not prompts:
        raise ValueError("at least one rendered prompt is required")
    if batch_size < 1 or max_length < 2:
        raise ValueError("batch_size and max_length must be positive")
    baseline_device = next(baseline.parameters()).device
    candidate_device = next(candidate.parameters()).device
    if baseline_device != candidate_device:
        raise ValueError("baseline and candidate must be on the same device")

    token_count = 0
    weighted_kl = 0.0
    weighted_top_mass = 0.0
    maximum_sequence_kl = 0.0
    for start in range(0, len(prompts), batch_size):
        batch = tokenizer(
            prompts[start : start + batch_size],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            return_token_type_ids=False,
        ).to(baseline_device)
        mask = batch["attention_mask"].bool()
        if not bool(mask.any(dim=1).all()):
            raise ValueError("every prompt must contain at least one token")
        positions = torch.arange(mask.shape[1], device=mask.device)[None, :]
        last_positions = positions.masked_fill(~mask, -1).amax(dim=1)
        rows = torch.arange(mask.shape[0], device=mask.device)
        mask[rows, last_positions] = False
        if not bool(mask.any()):
            raise ValueError("teacher-forced KL requires at least two tokens")
        baseline_logits = baseline(**batch, use_cache=False).logits.float()
        candidate_logits = candidate(**batch, use_cache=False).logits.float()
        drift = teacher_forced_sequence_kl(
            baseline_logits,
            candidate_logits,
            mask,
        )
        token_count += drift.token_count
        weighted_kl += drift.mean_token_kl * drift.token_count
        weighted_top_mass += drift.mean_topk_mass_coverage * drift.token_count
        maximum_sequence_kl = max(
            maximum_sequence_kl,
            drift.maximum_sequence_kl,
        )

    return SequenceDrift(
        token_count=token_count,
        mean_token_kl=weighted_kl / token_count,
        maximum_sequence_kl=maximum_sequence_kl,
        mean_topk_mass_coverage=weighted_top_mass / token_count,
    )


def certify_capability_preservation(
    baseline_slices: Mapping[str, Sequence[float] | Tensor],
    candidate_slices: Mapping[str, Sequence[float] | Tensor],
    margins: Mapping[str, float],
    sequence_drift: SequenceDrift,
    *,
    sequence_kl_maximum: float,
    alpha: float = 0.05,
    resamples: int = 10_000,
    seed: int = 0,
    require_equivalence: bool = False,
) -> CapabilityCertificate:
    """Apply simultaneous paired bounds across all preregistered slices."""

    names = tuple(sorted(margins))
    if not names or set(baseline_slices) != set(names) or set(candidate_slices) != set(names):
        raise ValueError("baseline, candidate, and preregistered margin slices must match exactly")
    if sequence_kl_maximum < 0 or not 0 < alpha < 1:
        raise ValueError("sequence KL maximum or alpha is invalid")
    # Bonferroni gives a simple auditable family-wise confidence guarantee.
    slice_alpha = alpha / len(names)
    intervals = {
        name: paired_bootstrap_interval(
            baseline_slices[name],
            candidate_slices[name],
            margin=margins[name],
            alpha=slice_alpha,
            resamples=resamples,
            seed=seed + index,
        )
        for index, name in enumerate(names)
    }
    blockers = []
    for name, interval in intervals.items():
        passed = interval.equivalence_passed if require_equivalence else interval.noninferiority_passed
        if not passed:
            blockers.append(f"slice:{name}")
    if sequence_drift.mean_token_kl > sequence_kl_maximum:
        blockers.append("sequence-kl")
    return CapabilityCertificate(
        passed=not blockers,
        simultaneous_alpha=alpha,
        slices=intervals,
        sequence_drift=sequence_drift,
        sequence_kl_maximum=sequence_kl_maximum,
        blocking_reasons=tuple(blockers),
    )


def certify_artifact_set(
    certificates: Mapping[str, CapabilityCertificate],
    *,
    required_artifacts: Sequence[str],
) -> ArtifactCapabilitySet:
    """Require BF16 and every distributed quantization to pass independently."""

    required = tuple(sorted(set(required_artifacts)))
    if not required:
        raise ValueError("at least one artifact must be preregistered")
    blockers = tuple(
        name
        for name in required
        if name not in certificates or not certificates[name].passed
    )
    return ArtifactCapabilitySet(
        passed=not blockers,
        required_artifacts=required,
        certificates=dict(certificates),
        blocking_artifacts=blockers,
    )
