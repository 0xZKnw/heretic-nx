"""Architecture-neutral residual-stream contrast estimation."""

from __future__ import annotations

import torch
from torch import Tensor

from .contrastive import ContrastiveAxis, fit_contrastive_axis


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
