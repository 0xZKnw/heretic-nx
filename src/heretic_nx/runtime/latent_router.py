"""Conservative latent harmfulness guard and task prototype router."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor
import torch.nn.functional as F


@dataclass(frozen=True)
class RouteDecision:
    action: Literal["route", "abstain-harmfulness", "abstain-task"]
    task: str | None
    harmfulness_score: float
    harmfulness_threshold: float
    task_similarity: float | None


@dataclass(frozen=True)
class LatentSafetyRouter:
    center: Tensor
    scale: Tensor
    harmfulness_axis: Tensor
    harmfulness_threshold: float
    task_labels: tuple[str, ...]
    task_centroids: Tensor
    minimum_task_similarity: float

    @classmethod
    def fit(
        cls,
        safe_instruction_states: Tensor,
        unsafe_instruction_states: Tensor,
        safe_task_labels: Sequence[str],
        *,
        unsafe_recall: float = 1.0,
        minimum_task_similarity: float | None = None,
        eps: float = 1e-6,
    ) -> "LatentSafetyRouter":
        """Fit a diagonal-LDA guard and safe-task prototypes.

        The threshold is chosen from unsafe calibration scores. At the default
        recall of 1.0 every observed unsafe calibration example makes the router
        abstain, even if this sacrifices some benign routing coverage.
        """

        safe = safe_instruction_states.float()
        unsafe = unsafe_instruction_states.float()
        if safe.ndim != 2 or unsafe.ndim != 2 or safe.shape[1] != unsafe.shape[1]:
            raise ValueError("safe and unsafe states must be compatible matrices")
        if safe.shape[0] < 2 or unsafe.shape[0] < 2:
            raise ValueError("at least two safe and unsafe states are required")
        if len(safe_task_labels) != safe.shape[0]:
            raise ValueError("safe_task_labels length must match safe states")
        if not 0 < unsafe_recall <= 1:
            raise ValueError("unsafe_recall must be in (0, 1]")

        all_states = torch.cat((safe, unsafe), dim=0)
        center = all_states.mean(dim=0)
        pooled_variance = all_states.var(dim=0, unbiased=False)
        variance_floor = pooled_variance.mean().clamp_min(eps) * 1e-3
        scale = pooled_variance.clamp_min(variance_floor).sqrt()
        safe_z = (safe - center) / scale
        unsafe_z = (unsafe - center) / scale
        axis = unsafe_z.mean(dim=0) - safe_z.mean(dim=0)
        axis = F.normalize(axis, dim=0, eps=eps)
        unsafe_scores = unsafe_z @ axis
        threshold_quantile = max(0.0, 1.0 - unsafe_recall)
        threshold = float(torch.quantile(unsafe_scores, threshold_quantile).item())
        threshold -= eps

        labels = tuple(sorted({str(label) for label in safe_task_labels}))
        normalized_safe = F.normalize(safe_z, dim=1, eps=eps)
        centroids = []
        own_similarities = []
        for label in labels:
            mask = torch.tensor(
                [str(item) == label for item in safe_task_labels],
                dtype=torch.bool,
                device=safe.device,
            )
            centroid = F.normalize(normalized_safe[mask].mean(dim=0), dim=0, eps=eps)
            centroids.append(centroid)
            own_similarities.append(normalized_safe[mask] @ centroid)
        centroid_matrix = torch.stack(centroids, dim=0)
        if minimum_task_similarity is None:
            training_similarity = torch.cat(own_similarities)
            minimum_task_similarity = float(torch.quantile(training_similarity, 0.05).item())

        return cls(
            center=center,
            scale=scale,
            harmfulness_axis=axis,
            harmfulness_threshold=threshold,
            task_labels=labels,
            task_centroids=centroid_matrix,
            minimum_task_similarity=float(minimum_task_similarity),
        )

    def harmfulness_scores(self, instruction_states: Tensor) -> Tensor:
        states = instruction_states.float()
        return ((states - self.center) / self.scale) @ self.harmfulness_axis

    def decide(self, instruction_state: Tensor) -> RouteDecision:
        if instruction_state.ndim != 1:
            raise ValueError("instruction_state must be one-dimensional")
        standardized = (instruction_state.float() - self.center) / self.scale
        score = float((standardized @ self.harmfulness_axis).item())
        if score >= self.harmfulness_threshold:
            return RouteDecision(
                "abstain-harmfulness",
                None,
                score,
                self.harmfulness_threshold,
                None,
            )
        similarities = self.task_centroids @ F.normalize(standardized, dim=0)
        best_index = int(similarities.argmax().item())
        best_similarity = float(similarities[best_index].item())
        if best_similarity < self.minimum_task_similarity:
            return RouteDecision(
                "abstain-task",
                None,
                score,
                self.harmfulness_threshold,
                best_similarity,
            )
        return RouteDecision(
            "route",
            self.task_labels[best_index],
            score,
            self.harmfulness_threshold,
            best_similarity,
        )


@dataclass(frozen=True)
class ConsensusSafetyRouter:
    """Conservative OR ensemble for architecture-diverse risk probes."""

    probes: Mapping[str, LatentSafetyRouter]
    task_site_id: str

    def __post_init__(self) -> None:
        if not self.probes:
            raise ValueError("at least one risk probe is required")
        if self.task_site_id not in self.probes:
            raise ValueError("task_site_id must reference a risk probe")

    def decide(self, instruction_states: Mapping[str, Tensor]) -> RouteDecision:
        missing = set(self.probes) - set(instruction_states)
        if missing:
            # Missing semantic readings are a routing failure, never permission to edit.
            return RouteDecision("abstain-task", None, float("nan"), float("nan"), None)
        decisions = {
            site_id: router.decide(instruction_states[site_id])
            for site_id, router in self.probes.items()
        }
        for site_id in sorted(decisions):
            decision = decisions[site_id]
            if decision.action == "abstain-harmfulness":
                return decision
        return decisions[self.task_site_id]
