"""Model-agnostic residual-stream weight editor construction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from heretic_nx.geometry.contrastive import ContrastiveAxis
from heretic_nx.model.semantic_sites import SemanticSite, SemanticSiteRegistry

from .activation_op import ActivationOperator
from .norm_preserving import norm_preserving_weight_edit


@dataclass(frozen=True)
class ResidualStreamWeightEditor:
    """A semantic projection site tied to its block residual direction."""

    site: SemanticSite
    operator: ActivationOperator
    evidence: ContrastiveAxis

    @property
    def site_id(self) -> str:
        return self.site.id

    @property
    def module_path(self) -> str:
        return self.site.module_path


def build_residual_stream_weight_editors(
    registry: SemanticSiteRegistry,
    axes: Sequence[ContrastiveAxis] | Mapping[int, ContrastiveAxis],
    *,
    families: frozenset[str] = frozenset({"gqa", "ffn", "liv"}),
) -> tuple[ResidualStreamWeightEditor, ...]:
    """Bind per-block residual axes to architecture-discovered output projections."""

    if not families:
        raise ValueError("at least one semantic family must be selected")
    axis_by_layer = dict(enumerate(axes)) if isinstance(axes, Sequence) else dict(axes)
    editors = []
    for site in registry.sites:
        if site.family not in families:
            continue
        evidence = axis_by_layer.get(site.layer)
        if evidence is None:
            raise ValueError(f"missing residual axis for selected layer {site.layer}")
        axis = evidence.axis.detach().float()
        if axis.ndim != 1 or axis.shape[0] != site.stream_dim:
            raise ValueError(f"residual axis dimension mismatch at {site.id}")
        if site.output_dim != site.stream_dim:
            raise ValueError(f"selected projection does not emit the residual width: {site.id}")
        column = axis[:, None].cpu()
        editors.append(
            ResidualStreamWeightEditor(
                site=site,
                operator=ActivationOperator(column, column, 1.0),
                evidence=evidence,
            )
        )
    if not editors:
        raise RuntimeError("semantic registry exposes no selected residual projections")
    return tuple(editors)


def snapshot_residual_stream_weights(
    model: nn.Module,
    editors: Sequence[ResidualStreamWeightEditor],
) -> dict[str, Tensor]:
    """Capture immutable CPU copies of every selected base weight."""

    return {
        editor.site_id: model.get_submodule(editor.module_path)
        .weight.detach()
        .cpu()
        .clone()
        for editor in editors
    }


@torch.no_grad()
def apply_residual_stream_weight_edits(
    model: nn.Module,
    editors: Sequence[ResidualStreamWeightEditor],
    originals: Mapping[str, Tensor],
    strengths: Mapping[str, float],
) -> None:
    """Apply a complete semantic portfolio, restoring zero-strength sites."""

    editor_ids = {editor.site_id for editor in editors}
    if set(originals) != editor_ids:
        raise ValueError("original weight keys must exactly match the editor portfolio")
    unknown = set(strengths) - editor_ids
    if unknown:
        raise ValueError(f"strengths contain unknown semantic sites: {sorted(unknown)}")
    for editor in editors:
        module = model.get_submodule(editor.module_path)
        weight = getattr(module, "weight", None)
        if weight is None or weight.ndim != 2:
            raise ValueError(f"selected module has no matrix weight: {editor.module_path}")
        base = originals[editor.site_id].to(weight.device)
        strength = float(strengths.get(editor.site_id, 0.0))
        edited = (
            norm_preserving_weight_edit(base, editor.operator, strength=strength)
            if strength > 0
            else base
        )
        weight.copy_(edited.to(weight.dtype))
