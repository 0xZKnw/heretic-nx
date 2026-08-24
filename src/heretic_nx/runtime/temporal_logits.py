"""Fail-closed generation control for overlong explicit thinking spans."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class TemporalCloseDecision:
    active_rows: tuple[int, ...]
    generated_tokens: int
    forced: bool


class TemporalThinkController:
    """Transformers-compatible logits processor for explicit ``<think>`` spans.

    It is intentionally inert unless both external routing gates have accepted
    the request. The controller first adds a bounded preference for the closing
    token and can force closure after a small grace interval. It is a runtime
    sidecar and never mutates model weights.
    """

    def __init__(
        self,
        *,
        prompt_length: int,
        open_token_id: int,
        close_token_id: int,
        budget_tokens: int = 96,
        grace_tokens: int = 8,
        close_logit_boost: float = 12.0,
        risk_gate_passed: bool,
        task_route_passed: bool,
    ) -> None:
        if prompt_length < 0:
            raise ValueError("prompt_length must be non-negative")
        if min(open_token_id, close_token_id) < 0:
            raise ValueError("think token identifiers must be non-negative")
        if budget_tokens < 1 or grace_tokens < 0:
            raise ValueError("token budgets are invalid")
        if close_logit_boost < 0:
            raise ValueError("close_logit_boost must be non-negative")
        self.prompt_length = prompt_length
        self.open_token_id = open_token_id
        self.close_token_id = close_token_id
        self.budget_tokens = budget_tokens
        self.grace_tokens = grace_tokens
        self.close_logit_boost = close_logit_boost
        self.enabled = bool(risk_gate_passed and task_route_passed)
        self.last_decision = TemporalCloseDecision((), 0, False)

    def _row_is_open(self, row: Tensor) -> bool:
        generated = row[self.prompt_length :]
        opens = torch.nonzero(generated == self.open_token_id, as_tuple=False).flatten()
        if opens.numel() == 0:
            return False
        last_open = int(opens[-1].item())
        return not bool(torch.any(generated[last_open + 1 :] == self.close_token_id).item())

    def __call__(self, input_ids: Tensor, scores: Tensor) -> Tensor:
        if input_ids.ndim != 2 or scores.ndim != 2 or input_ids.shape[0] != scores.shape[0]:
            raise ValueError("input_ids and scores must be aligned rank-2 tensors")
        if input_ids.device != scores.device:
            raise ValueError("input_ids and scores must be on the same device")
        if self.close_token_id >= scores.shape[1]:
            raise ValueError("close_token_id is outside the score vocabulary")
        generated_tokens = max(input_ids.shape[1] - self.prompt_length, 0)
        if not self.enabled or generated_tokens < self.budget_tokens:
            self.last_decision = TemporalCloseDecision((), generated_tokens, False)
            return scores

        rows = tuple(index for index, row in enumerate(input_ids) if self._row_is_open(row))
        if not rows:
            self.last_decision = TemporalCloseDecision((), generated_tokens, False)
            return scores

        result = scores.clone()
        # ``generated_tokens`` is counted before the next token is sampled. Force
        # one step early so the emitted close token itself stays inside the budget
        # plus grace interval (no off-by-one promotion failure).
        force = generated_tokens >= self.budget_tokens + max(self.grace_tokens, 1) - 1
        for row in rows:
            if force:
                result[row].fill_(-torch.inf)
                result[row, self.close_token_id] = 0.0
            else:
                result[row, self.close_token_id] += self.close_logit_boost
        self.last_decision = TemporalCloseDecision(rows, generated_tokens, force)
        return result
