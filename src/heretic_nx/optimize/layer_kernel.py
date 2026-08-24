"""Auditable smooth layer-strength kernels for static interventions."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class LayerKernel:
    """Piecewise-linear strength profile centered on a fractional layer."""

    maximum_strength: float
    center: float
    minimum_strength: float
    radius: float

    def __post_init__(self) -> None:
        values = (
            self.maximum_strength,
            self.center,
            self.minimum_strength,
            self.radius,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("layer-kernel parameters must be finite")
        if self.maximum_strength < 0 or self.minimum_strength < 0:
            raise ValueError("layer-kernel strengths must be non-negative")
        if self.minimum_strength > self.maximum_strength:
            raise ValueError("minimum strength cannot exceed maximum strength")
        if self.radius <= 0:
            raise ValueError("layer-kernel radius must be positive")

    def strength(self, layer: int) -> float:
        if layer < 0:
            raise ValueError("layer index must be non-negative")
        distance = abs(float(layer) - self.center)
        if distance > self.radius:
            return 0.0
        fraction = distance / self.radius
        return self.maximum_strength + fraction * (
            self.minimum_strength - self.maximum_strength
        )

    def active_layers(self, layer_count: int) -> tuple[int, ...]:
        if layer_count < 1:
            raise ValueError("layer_count must be positive")
        return tuple(
            layer for layer in range(layer_count) if self.strength(layer) > 0
        )
