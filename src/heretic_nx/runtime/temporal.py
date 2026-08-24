"""Fail-closed temporal gates and bounded feedback control."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TemporalGate:
    budget_tokens: int = 96
    maximum_window_tokens: int = 32
    activation_threshold: float = 0.5
    risk_threshold: float = 0.5
    tokens_used: int = 0
    window_used: int = 0
    active: bool = False
    shut_down: bool = False

    def step(self, *, task_score: float, risk_score: float, checkpoint: bool = True) -> float:
        if self.shut_down:
            return 0.0
        if risk_score >= self.risk_threshold:
            self.active = False
            self.shut_down = True
            return 0.0
        if self.tokens_used >= self.budget_tokens:
            return 0.0
        if checkpoint and task_score >= self.activation_threshold:
            self.active = True
            self.window_used = 0
        if not self.active:
            return 0.0
        if self.window_used >= self.maximum_window_tokens:
            self.active = False
            return 0.0
        self.tokens_used += 1
        self.window_used += 1
        return 1.0


@dataclass
class BoundedPIDController:
    proportional: float
    integral: float
    derivative: float
    beta_max: float
    rate_limit: float
    integral_limit: float
    _integral_state: float = 0.0
    _previous_error: float = 0.0
    _previous_beta: float = 0.0

    def step(self, error: float, *, risk_shutdown: bool = False) -> float:
        if risk_shutdown:
            self._integral_state = 0.0
            self._previous_beta = 0.0
            self._previous_error = error
            return 0.0
        candidate_integral = max(
            -self.integral_limit,
            min(self.integral_limit, self._integral_state + error),
        )
        raw = (
            self.proportional * error
            + self.integral * candidate_integral
            + self.derivative * (error - self._previous_error)
        )
        clipped = max(0.0, min(self.beta_max, raw))
        lower = max(0.0, self._previous_beta - self.rate_limit)
        upper = min(self.beta_max, self._previous_beta + self.rate_limit)
        beta = max(lower, min(upper, clipped))
        # Anti-windup: accept integral growth only while the output is not saturated.
        if 0.0 < clipped < self.beta_max:
            self._integral_state = candidate_integral
        self._previous_error = error
        self._previous_beta = beta
        return beta
