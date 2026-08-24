"""Verified benign minimal-pair records and paired activation differences."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from .oracles import OracleVerdict, benign_consensus


@dataclass(frozen=True)
class BenignPromptPair:
    pair_id: str
    task: str
    anchor: str
    contrast: str
    transformation: str
    anchor_oracles: tuple[OracleVerdict, ...]
    contrast_oracles: tuple[OracleVerdict, ...]

    def validated(self, *, confidence_minimum: float = 0.8) -> bool:
        return (
            bool(self.anchor.strip())
            and bool(self.contrast.strip())
            and self.anchor != self.contrast
            and benign_consensus(self.anchor_oracles, confidence_minimum=confidence_minimum)
            and benign_consensus(self.contrast_oracles, confidence_minimum=confidence_minimum)
        )


def paired_differences(anchor_activations: Tensor, contrast_activations: Tensor) -> Tensor:
    if anchor_activations.shape != contrast_activations.shape or anchor_activations.ndim != 2:
        raise ValueError("paired activations must be aligned matrices")
    return contrast_activations - anchor_activations
