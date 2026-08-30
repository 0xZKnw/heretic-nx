"""Paired capability preservation and teacher-forced sequence drift metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import Tensor


# Keep each floating-point row buffer near 64 MiB for float32 logits.  The KL
# kernel uses at most a small fixed number of these buffers instead of
# materializing full [batch, token, vocabulary] probability tensors.
_KL_CHUNK_TARGET_ELEMENTS = 16 * 1024 * 1024


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

    def __post_init__(self) -> None:
        if (
            isinstance(self.token_count, bool)
            or not isinstance(self.token_count, (int, np.integer))
            or self.token_count < 1
        ):
            raise ValueError("sequence drift token_count must be positive")
        metrics = (
            self.mean_token_kl,
            self.maximum_sequence_kl,
            self.mean_topk_mass_coverage,
        )
        if any(not np.isfinite(value) for value in metrics):
            raise ValueError("sequence drift metrics must be finite")
        if self.mean_token_kl < -1e-6 or self.maximum_sequence_kl < -1e-6:
            raise ValueError("sequence KL metrics cannot be negative")
        if self.maximum_sequence_kl + 1e-6 < self.mean_token_kl:
            raise ValueError("maximum sequence KL cannot be below mean token KL")
        if not 0.0 <= self.mean_topk_mass_coverage <= 1.0 + 1e-6:
            raise ValueError("top-k mass coverage must be in [0, 1]")


@dataclass(frozen=True)
class CapabilityCertificate:
    passed: bool
    simultaneous_alpha: float
    slices: dict[str, PairedInterval]
    sequence_drift: SequenceDrift
    sequence_kl_maximum: float
    blocking_reasons: tuple[str, ...]
    artifact_id: str | None = None
    artifact_sha256: str | None = None
    quantization: str | None = None

    def __post_init__(self) -> None:
        if (
            not np.isfinite(self.simultaneous_alpha)
            or not 0 < self.simultaneous_alpha < 1
            or not np.isfinite(self.sequence_kl_maximum)
            or self.sequence_kl_maximum < 0
        ):
            raise ValueError("certificate confidence and KL threshold must be finite")
        if self.passed != (not self.blocking_reasons):
            raise ValueError("certificate passed flag is inconsistent with blockers")
        sequence_blocked = "sequence-kl" in self.blocking_reasons
        if sequence_blocked != (
            self.sequence_drift.mean_token_kl > self.sequence_kl_maximum
        ):
            raise ValueError("certificate sequence KL blocker is inconsistent")
        for name, interval in self.slices.items():
            values = (
                interval.mean_difference,
                interval.lower,
                interval.upper,
                interval.margin,
            )
            if interval.count < 2 or any(not np.isfinite(value) for value in values):
                raise ValueError(f"certificate slice {name!r} is invalid")
            if (
                not interval.noninferiority_passed
                and f"slice:{name}" not in self.blocking_reasons
            ):
                raise ValueError(f"certificate slice blocker is missing for {name!r}")
        identity = (self.artifact_id, self.artifact_sha256, self.quantization)
        supplied = tuple(value is not None for value in identity)
        if not any(supplied):
            # Legacy construction remains possible, but an unbound certificate
            # cannot promote an artifact set.
            return
        if not all(supplied):
            raise ValueError(
                "artifact_id, artifact_sha256, and quantization must be supplied together"
            )
        if not self.artifact_id or not self.artifact_id.strip():
            raise ValueError("artifact_id must be non-empty")
        if (
            not self.artifact_sha256
            or len(self.artifact_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.artifact_sha256
            )
        ):
            raise ValueError("artifact_sha256 must be a lowercase SHA-256 digest")
        if not self.quantization or not self.quantization.strip():
            raise ValueError("quantization must be non-empty")

    @property
    def artifact_bound(self) -> bool:
        """Whether this certificate names one immutable deployment artifact."""

        return self.artifact_id is not None


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
    if (
        not np.isfinite(margin)
        or not np.isfinite(alpha)
        or margin < 0
        or not 0 < alpha < 1
        or resamples < 100
    ):
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


@torch.inference_mode()
def teacher_forced_sequence_kl(
    baseline_logits: Tensor,
    candidate_logits: Tensor,
    token_mask: Tensor,
    *,
    top_k: int = 128,
    token_chunk_size: int | None = None,
) -> SequenceDrift:
    """Compute full-vocabulary KL on every selected teacher-forced token.

    Padding and other unselected positions are never passed through softmax.
    Selected rows are processed in bounded chunks and the two log-probability
    buffers are reused in place.  This keeps the calculation exact while
    avoiding several full ``[batch, token, vocabulary]`` intermediates.

    ``token_chunk_size=None`` chooses a vocabulary-aware working set.  The
    optional override is primarily useful for constrained evaluators and
    reproducibility tests; it does not change the resulting metric.
    """

    if (
        baseline_logits.ndim != 3
        or candidate_logits.shape != baseline_logits.shape
    ):
        raise ValueError("baseline and candidate logits must be aligned [batch, token, vocab]")
    if token_mask.shape != baseline_logits.shape[:2]:
        raise ValueError("token_mask must align with batch and token dimensions")
    if (
        baseline_logits.device != candidate_logits.device
        or baseline_logits.device != token_mask.device
    ):
        raise ValueError("sequence KL tensors must be on the same device")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be positive")
    if token_chunk_size is not None and (
        isinstance(token_chunk_size, bool)
        or not isinstance(token_chunk_size, int)
        or token_chunk_size < 1
    ):
        raise ValueError("token_chunk_size must be positive")
    vocabulary_size = baseline_logits.shape[-1]
    if vocabulary_size < 1:
        raise ValueError("sequence KL vocabulary must be non-empty")
    top_k = min(top_k, vocabulary_size)
    mask = token_mask.bool()
    if not mask.any():
        raise ValueError("token_mask must select at least one token")

    # ``isfinite`` over a giant logits tensor allocates a same-shaped boolean
    # temporary.  Finite extrema provide the same fail-closed validation with
    # scalar outputs and no vocabulary-sized allocation.
    base_minimum, base_maximum = torch.aminmax(baseline_logits)
    candidate_minimum, candidate_maximum = torch.aminmax(candidate_logits)
    finite_extrema = (
        torch.isfinite(base_minimum)
        & torch.isfinite(base_maximum)
        & torch.isfinite(candidate_minimum)
        & torch.isfinite(candidate_maximum)
    )
    if not bool(finite_extrema):
        raise ValueError("sequence KL logits must be finite")

    if token_chunk_size is None:
        token_chunk_size = max(1, _KL_CHUNK_TARGET_ELEMENTS // vocabulary_size)

    # ``nonzero`` is row-major, so chunks retain sequence and token order.  Use
    # two-dimensional advanced indexing instead of reshaping logits: the latter
    # could silently copy an entire non-contiguous model output.
    selected_positions = mask.nonzero(as_tuple=False)
    selected_count = selected_positions.shape[0]
    selected_kl = torch.empty(
        selected_count,
        device=baseline_logits.device,
        dtype=torch.float32,
    )
    selected_top_mass = torch.empty_like(selected_kl)
    for start in range(0, selected_count, token_chunk_size):
        stop = min(start + token_chunk_size, selected_count)
        positions = selected_positions[start:stop]
        rows = positions[:, 0]
        tokens = positions[:, 1]
        base_log_probs = torch.log_softmax(
            baseline_logits[rows, tokens].float(),
            dim=-1,
        )
        candidate_log_probs = torch.log_softmax(
            candidate_logits[rows, tokens].float(),
            dim=-1,
        )

        # Top-k values are sufficient for mass coverage; retaining indices and
        # gathering from a separate full probability tensor is unnecessary.
        selected_top_mass[start:stop] = (
            torch.topk(base_log_probs, k=top_k, dim=-1)
            .values.exp()
            .sum(dim=-1)
        )

        # Reuse the two private selected-row buffers: candidate_log_probs holds
        # log(p)-log(q), then base_log_probs becomes p.
        candidate_log_probs.neg_().add_(base_log_probs)
        base_log_probs.exp_()
        selected_kl[start:stop] = torch.sum(
            base_log_probs * candidate_log_probs,
            dim=-1,
        )

    counts = mask.sum(dim=1)
    sequence_means = torch.stack(
        [
            values.mean()
            for values in selected_kl.split(counts.tolist())
            if values.numel() > 0
        ]
    )
    return SequenceDrift(
        token_count=selected_count,
        mean_token_kl=float(selected_kl.mean().item()),
        maximum_sequence_kl=float(sequence_means.max().item()),
        mean_topk_mass_coverage=float(selected_top_mass.mean().item()),
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
    artifact_id: str | None = None,
    artifact_sha256: str | None = None,
    quantization: str | None = None,
) -> CapabilityCertificate:
    """Apply simultaneous paired bounds across all preregistered slices.

    Supplying all three artifact identity fields binds the result to the exact
    evaluated bytes. Legacy callers may omit all three, but unbound results are
    intentionally ineligible for :func:`certify_artifact_set`.
    """

    names = tuple(sorted(margins))
    if (
        not names
        or set(baseline_slices) != set(names)
        or set(candidate_slices) != set(names)
    ):
        raise ValueError(
            "baseline, candidate, and preregistered margin slices must match exactly"
        )
    if (
        not np.isfinite(sequence_kl_maximum)
        or not np.isfinite(alpha)
        or sequence_kl_maximum < 0
        or not 0 < alpha < 1
    ):
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
        passed = (
            interval.equivalence_passed
            if require_equivalence
            else interval.noninferiority_passed
        )
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
        artifact_id=artifact_id,
        artifact_sha256=artifact_sha256,
        quantization=quantization,
    )


def certify_artifact_set(
    certificates: Mapping[str, CapabilityCertificate],
    *,
    required_artifacts: Sequence[str] | Mapping[str, str],
) -> ArtifactCapabilitySet:
    """Require every distributed artifact to pass under its own identity.

    ``required_artifacts`` accepts the historical sequence of artifact names,
    where each name is also treated as the expected quantization label, or a
    mapping from artifact ID to expected quantization. Certificates must be
    bound to a lowercase SHA-256 digest; a digest cannot certify two artifacts.
    """

    if isinstance(required_artifacts, Mapping):
        expected_quantizations = dict(required_artifacts)
    else:
        if isinstance(required_artifacts, (str, bytes)):
            raise ValueError("required_artifacts must be a sequence of artifact IDs")
        expected_quantizations = {name: name for name in required_artifacts}
    if any(
        not isinstance(name, str)
        or not name.strip()
        or not isinstance(quantization, str)
        or not quantization.strip()
        for name, quantization in expected_quantizations.items()
    ):
        raise ValueError("artifact IDs and quantization labels must be non-empty strings")
    required = tuple(sorted(expected_quantizations))
    if not required:
        raise ValueError("at least one artifact must be preregistered")

    blockers: set[str] = set()
    valid_identities: dict[str, CapabilityCertificate] = {}
    for name in required:
        certificate = certificates.get(name)
        if certificate is None or not certificate.passed or not certificate.artifact_bound:
            blockers.add(name)
            continue
        if certificate.artifact_id != name:
            blockers.add(name)
            continue
        expected_quantization = expected_quantizations[name].strip().casefold()
        if (
            certificate.quantization is None
            or certificate.quantization.strip().casefold() != expected_quantization
        ):
            blockers.add(name)
            continue
        valid_identities[name] = certificate

    names_by_digest: dict[str, list[str]] = {}
    for name, certificate in valid_identities.items():
        assert certificate.artifact_sha256 is not None
        names_by_digest.setdefault(certificate.artifact_sha256, []).append(name)
    for names in names_by_digest.values():
        if len(names) > 1:
            blockers.update(names)

    return ArtifactCapabilitySet(
        passed=not blockers,
        required_artifacts=required,
        certificates=dict(certificates),
        blocking_artifacts=tuple(name for name in required if name in blockers),
    )
