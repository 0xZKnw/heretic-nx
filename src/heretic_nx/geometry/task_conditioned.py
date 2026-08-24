"""Task-conditioned contrasts for benign over-refusal."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from .principal_angles import orthonormal_basis


@dataclass(frozen=True)
class TaskContrast:
    label: str
    refused_count: int
    answered_count: int
    direction: Tensor


@dataclass(frozen=True)
class TaskConditionedGeometry:
    contrasts: tuple[TaskContrast, ...]
    pooled_basis: Tensor


def fit_task_conditioned_geometry(
    activations: Tensor,
    task_labels: Sequence[str],
    refused: Tensor | Sequence[bool],
    *,
    minimum_per_class: int = 2,
) -> TaskConditionedGeometry:
    """Fit within-task ``benign-refused - benign-answered`` directions.

    Pooling raw tasks would mostly recover task identity. The pooled basis here is
    built only after each task's semantic centroid has cancelled out.
    """

    if activations.ndim != 2:
        raise ValueError("activations must have shape (samples, features)")
    if len(task_labels) != activations.shape[0]:
        raise ValueError("task_labels length must match activations")
    refused_mask = torch.as_tensor(refused, dtype=torch.bool, device=activations.device)
    if refused_mask.ndim != 1 or refused_mask.shape[0] != activations.shape[0]:
        raise ValueError("refused must have one value per activation")
    if minimum_per_class < 1:
        raise ValueError("minimum_per_class must be positive")

    indices: dict[str, list[int]] = defaultdict(list)
    for index, label in enumerate(task_labels):
        indices[str(label)].append(index)

    contrasts: list[TaskContrast] = []
    for label in sorted(indices):
        task_index = torch.tensor(indices[label], device=activations.device)
        task_values = activations.index_select(0, task_index)
        task_refused = refused_mask.index_select(0, task_index)
        positive = task_values[task_refused]
        negative = task_values[~task_refused]
        if positive.shape[0] < minimum_per_class or negative.shape[0] < minimum_per_class:
            continue
        direction = positive.mean(dim=0) - negative.mean(dim=0)
        norm = direction.norm()
        if not torch.isfinite(norm) or float(norm) <= torch.finfo(direction.dtype).eps:
            continue
        contrasts.append(
            TaskContrast(
                label=label,
                refused_count=positive.shape[0],
                answered_count=negative.shape[0],
                direction=direction / norm,
            )
        )

    if not contrasts:
        empty = activations.new_empty((activations.shape[1], 0))
        return TaskConditionedGeometry((), empty)
    pooled = orthonormal_basis(torch.stack([item.direction for item in contrasts], dim=1))
    return TaskConditionedGeometry(tuple(contrasts), pooled)
