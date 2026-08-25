"""Architecture-neutral residual-stream contrast estimation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from .contrastive import ContrastiveAxis, fit_contrastive_axis


@dataclass(frozen=True)
class CapabilityProtectedResidualAxis:
    """Residual refusal axis after removing dominant benign capabilities."""

    evidence: ContrastiveAxis
    capability_basis: Tensor
    capability_singular_values: Tensor
    retained_fraction: float
    safe_projection_rms: float
    target_separation: float
    efficiency: float


def last_token_residual_stack(
    hidden_states: tuple[Tensor, ...] | list[Tensor],
    attention_mask: Tensor,
    *,
    exclude_embedding: bool = True,
) -> Tensor:
    """Stack the final non-padding residual for every transformer block.

    Hugging Face causal decoders conventionally return the embedding residual
    first and one residual after each block.  The mask-based position lookup is
    valid for both left- and right-padded batches.
    """

    states = tuple(hidden_states)
    if exclude_embedding:
        states = states[1:]
    if not states:
        raise ValueError("hidden_states must contain at least one block residual")
    if attention_mask.ndim != 2:
        raise ValueError("attention_mask must have shape (batch, sequence)")
    if not bool((attention_mask != 0).any(dim=1).all()):
        raise ValueError("every row must contain at least one unmasked token")
    batch, sequence = attention_mask.shape
    positions = torch.arange(sequence, device=attention_mask.device)[None, :]
    last_positions = positions.masked_fill(attention_mask == 0, -1).amax(dim=1)
    batch_positions = torch.arange(batch, device=attention_mask.device)
    residuals = []
    for state in states:
        if state.ndim != 3 or state.shape[:2] != (batch, sequence):
            raise ValueError(
                "each hidden state must have shape (batch, sequence, dimension)"
            )
        residuals.append(state[batch_positions, last_positions])
    stacked = torch.stack(residuals, dim=1)
    if not torch.isfinite(stacked).all():
        raise ValueError("residual stack must be finite")
    return stacked


def fit_residual_stream_axes(
    safe: Tensor,
    target: Tensor,
    *,
    folds: int = 3,
    remove_safe_mean: bool = True,
) -> tuple[ContrastiveAxis, ...]:
    """Fit one contrastive direction per residual-stream block."""

    if safe.ndim != 3 or target.ndim != 3 or safe.shape != target.shape:
        raise ValueError(
            "safe and target residuals must share (example, layer, dimension)"
        )
    return tuple(
        fit_contrastive_axis(
            safe[:, layer],
            target[:, layer],
            folds=folds,
            remove_safe_mean=remove_safe_mean,
        )
        for layer in range(safe.shape[1])
    )


def protect_residual_stream_axes(
    safe: Tensor,
    target: Tensor,
    axes: tuple[ContrastiveAxis, ...] | list[ContrastiveAxis],
    *,
    capability_rank: int = 8,
    oversample: int = 4,
    niter: int = 3,
    seed: int = 2600,
    device: str | torch.device = "cpu",
    tolerance: float = 1e-7,
) -> tuple[CapabilityProtectedResidualAxis, ...]:
    """Residualize each refusal axis against a low-rank benign subspace.

    The contrastive estimator remains the Residual-Stream core. PRIME adds a
    deterministic per-layer benign PCA and removes that protected span before
    intervention. Returned diagnostics support sparse efficiency routing.
    """

    if safe.ndim != 3 or target.ndim != 3 or safe.shape != target.shape:
        raise ValueError(
            "safe and target residuals must share (example, layer, dimension)"
        )
    if len(axes) != safe.shape[1]:
        raise ValueError("one contrastive axis is required per residual layer")
    if capability_rank <= 0 or oversample < 0 or niter < 0:
        raise ValueError("PCA parameters must be non-negative and rank positive")
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    maximum_rank = min(safe.shape[0] - 1, safe.shape[2])
    if capability_rank > maximum_rank:
        raise ValueError("capability rank exceeds the residual sample rank")

    compute_device = torch.device(device)
    protected = []
    for layer, source_axis in enumerate(axes):
        safe_values = safe[:, layer].float()
        target_values = target[:, layer].float()
        centered = safe_values - safe_values.mean(dim=0)
        q = min(capability_rank + oversample, maximum_rank)
        fork_devices = (
            [compute_device.index or 0]
            if compute_device.type == "cuda"
            else []
        )
        with torch.random.fork_rng(devices=fork_devices):
            torch.manual_seed(seed + layer)
            _u, singular, vectors = torch.pca_lowrank(
                centered.to(compute_device),
                q=q,
                center=False,
                niter=niter,
            )
        capability = vectors[:, :capability_rank].float().cpu()
        singular = singular[:capability_rank].float().cpu()
        raw = source_axis.axis.detach().float().cpu()
        residual = raw - capability @ (capability.T @ raw)
        residual_norm = torch.linalg.vector_norm(residual)
        if float(residual_norm) <= tolerance:
            raise RuntimeError(f"protected residual axis collapsed at layer {layer}")
        unit = F.normalize(residual, dim=0)
        safe_centered = safe_values.cpu() - safe_values.cpu().mean(dim=0)
        safe_coefficients = safe_centered @ unit
        target_delta = target_values.cpu().mean(dim=0) - safe_values.cpu().mean(dim=0)
        safe_rms = float(safe_coefficients.square().mean().sqrt())
        separation = abs(float(torch.dot(target_delta, unit)))
        efficiency = separation / max(safe_rms, tolerance)
        safe_mean = safe_values.cpu().mean(dim=0)
        safe_mean_norm = torch.linalg.vector_norm(safe_mean)
        safe_mean_cosine = (
            float(torch.dot(unit, safe_mean / safe_mean_norm))
            if float(safe_mean_norm) > tolerance
            else 0.0
        )
        protected.append(
            CapabilityProtectedResidualAxis(
                evidence=ContrastiveAxis(
                    axis=unit,
                    fold_cosine_minimum=source_axis.fold_cosine_minimum,
                    fold_cosine_mean=source_axis.fold_cosine_mean,
                    safe_mean_cosine=safe_mean_cosine,
                    folds=source_axis.folds,
                ),
                capability_basis=capability,
                capability_singular_values=singular,
                retained_fraction=float(residual_norm.square()),
                safe_projection_rms=safe_rms,
                target_separation=separation,
                efficiency=efficiency,
            )
        )
    return tuple(protected)
