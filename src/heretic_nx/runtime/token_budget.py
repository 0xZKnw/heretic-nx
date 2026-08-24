"""Independent batch/token/memory control for heterogeneous operations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .memory_controller import AIMDMemoryController, MemoryDecision


Operation = Literal[
    "sketch",
    "prefill",
    "decode",
    "backward",
    "hook",
    "lowrank",
    "logits",
    "generate",
]


def cuda_memory_caps(free_vram_bytes: int) -> tuple[int, int]:
    gib = 1024**3
    soft = min(int(6.25 * gib), int(0.78 * free_vram_bytes))
    hard = int(0.90 * free_vram_bytes)
    return soft, hard


@dataclass
class OperationBudget:
    token_budget: int
    batch_controller: AIMDMemoryController
    peak_bytes: int = 0
    calls: int = 0

    def batch_for(self, sequence_tokens: int) -> int:
        if sequence_tokens < 1:
            raise ValueError("sequence_tokens must be positive")
        token_limited = max(1, self.token_budget // sequence_tokens)
        return min(self.batch_controller.batch_size, token_limited)

    def observe(self, peak_bytes: int, free_margin_bytes: int) -> MemoryDecision:
        self.calls += 1
        self.peak_bytes = max(self.peak_bytes, peak_bytes)
        return self.batch_controller.observe(free_margin_bytes)


@dataclass
class OperationBudgetRegistry:
    budgets: dict[Operation, OperationBudget] = field(default_factory=dict)

    @classmethod
    def defaults(cls) -> "OperationBudgetRegistry":
        settings = {
            "sketch": (8192, 8),
            "prefill": (4096, 4),
            "decode": (512, 8),
            "backward": (2048, 1),
            "hook": (4096, 4),
            "lowrank": (16384, 16),
            "logits": (4096, 4),
            "generate": (2048, 1),
        }
        return cls(
            {
                operation: OperationBudget(
                    token_budget=tokens,
                    batch_controller=AIMDMemoryController(batch_size=batch),
                )
                for operation, (tokens, batch) in settings.items()
            }
        )

    def __getitem__(self, operation: Operation) -> OperationBudget:
        return self.budgets[operation]
