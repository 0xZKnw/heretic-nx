"""Independent AIMD batch controller for each runtime operation."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class MemoryDecision:
    batch_size: int
    retry: bool
    reason: str


@dataclass
class AIMDMemoryController:
    batch_size: int = 1
    minimum_batch: int = 1
    maximum_batch: int = 128
    target_margin_bytes: int = 1024**3
    stable_calls_before_growth: int = 3
    _stable_calls: int = field(default=0, init=False)
    _oom_at_batch: dict[int, int] = field(default_factory=dict, init=False)

    def observe(self, free_margin_bytes: int) -> MemoryDecision:
        if free_margin_bytes >= self.target_margin_bytes:
            self._stable_calls += 1
            if self._stable_calls >= self.stable_calls_before_growth:
                self.batch_size = min(self.maximum_batch, self.batch_size + 1)
                self._stable_calls = 0
                return MemoryDecision(self.batch_size, False, "additive-increase")
        else:
            self._stable_calls = 0
            reduced = max(self.minimum_batch, int(self.batch_size * 0.8))
            if reduced == self.batch_size and self.batch_size > self.minimum_batch:
                reduced -= 1
            self.batch_size = reduced
            return MemoryDecision(self.batch_size, False, "low-margin-decrease")
        return MemoryDecision(self.batch_size, False, "stable")

    def on_oom(self) -> MemoryDecision:
        failures = self._oom_at_batch.get(self.batch_size, 0) + 1
        self._oom_at_batch[self.batch_size] = failures
        if failures >= 2:
            return MemoryDecision(self.batch_size, False, "repeated-oom-fail")
        self.batch_size = max(self.minimum_batch, self.batch_size // 2)
        self._stable_calls = 0
        return MemoryDecision(self.batch_size, True, "oom-halving-retry")
