"""Architecture-aware semantic intervention-site discovery."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from torch import nn

from heretic_nx.hashing import sha256_json


SiteFamily = Literal["residual", "gqa", "ffn", "liv", "block"]
SiteKind = Literal["residual_out", "attention_out", "ffn_out", "liv_mix_out", "block_out"]


@dataclass(frozen=True)
class SemanticSite:
    id: str
    layer: int
    family: SiteFamily
    kind: SiteKind
    module_path: str
    module_type: str
    stream_dim: int
    input_dim: int | None
    output_dim: int | None
    structure_hash: str


@dataclass(frozen=True)
class SemanticSiteRegistry:
    sites: tuple[SemanticSite, ...]
    structure_hash: str

    def by_family(self, family: SiteFamily) -> tuple[SemanticSite, ...]:
        return tuple(site for site in self.sites if site.family == family)

    def by_kind(self, kind: SiteKind) -> tuple[SemanticSite, ...]:
        return tuple(site for site in self.sites if site.kind == kind)


def _model_layers(model: nn.Module) -> tuple[str, list[nn.Module]]:
    candidates = (
        ("model.layers", getattr(getattr(model, "model", None), "layers", None)),
        ("layers", getattr(model, "layers", None)),
    )
    for prefix, layers in candidates:
        if layers is not None and isinstance(layers, (nn.ModuleList, list, tuple)):
            return prefix, list(layers)
    raise ValueError("model does not expose a supported decoder-layer collection")


def _linear_shape(module: nn.Module) -> tuple[int | None, int | None]:
    input_dim = getattr(module, "in_features", None)
    output_dim = getattr(module, "out_features", None)
    if input_dim is not None and output_dim is not None:
        return int(input_dim), int(output_dim)
    weight = getattr(module, "weight", None)
    if weight is not None and getattr(weight, "ndim", 0) == 2:
        return int(weight.shape[1]), int(weight.shape[0])
    return None, None


def _module_signature(module: nn.Module) -> dict[str, object]:
    children = []
    for name, child in module.named_modules():
        input_dim, output_dim = _linear_shape(child)
        children.append(
            {
                "name": name or ".",
                "type": type(child).__name__,
                "input_dim": input_dim,
                "output_dim": output_dim,
            }
        )
    return {"type": type(module).__name__, "children": children}


def _site(
    *,
    layer: int,
    family: SiteFamily,
    kind: SiteKind,
    path: str,
    module: nn.Module,
    stream_dim: int,
) -> SemanticSite:
    input_dim, output_dim = _linear_shape(module)
    signature = _module_signature(module)
    return SemanticSite(
        id=f"L{layer:02d}:{kind}",
        layer=layer,
        family=family,
        kind=kind,
        module_path=path,
        module_type=type(module).__name__,
        stream_dim=stream_dim,
        input_dim=input_dim,
        output_dim=output_dim,
        structure_hash=sha256_json(signature),
    )


def discover_semantic_sites(model: nn.Module) -> SemanticSiteRegistry:
    """Discover LIV/GQA/FFN/block sites from module structure, not path regexes."""

    prefix, layers = _model_layers(model)
    configured_dim = getattr(getattr(model, "config", None), "hidden_size", None)
    sites: list[SemanticSite] = []
    for layer_index, layer in enumerate(layers):
        feed_forward = getattr(layer, "feed_forward", None)
        stream_dim = int(configured_dim or 0)
        if feed_forward is not None and hasattr(feed_forward, "w2"):
            _ffn_in, ffn_out = _linear_shape(feed_forward.w2)
            stream_dim = int(ffn_out or stream_dim)
        if stream_dim <= 0:
            raise ValueError(f"cannot infer stream dimension at layer {layer_index}")

        attention = getattr(layer, "self_attn", None)
        if attention is not None and all(
            hasattr(attention, name) for name in ("q_proj", "k_proj", "v_proj", "out_proj")
        ):
            sites.append(
                _site(
                    layer=layer_index,
                    family="gqa",
                    kind="attention_out",
                    path=f"{prefix}.{layer_index}.self_attn.out_proj",
                    module=attention.out_proj,
                    stream_dim=stream_dim,
                )
            )

        convolution = getattr(layer, "conv", None)
        if convolution is not None and all(
            hasattr(convolution, name) for name in ("conv", "in_proj", "out_proj")
        ):
            sites.append(
                _site(
                    layer=layer_index,
                    family="liv",
                    kind="liv_mix_out",
                    path=f"{prefix}.{layer_index}.conv.out_proj",
                    module=convolution.out_proj,
                    stream_dim=stream_dim,
                )
            )

        if feed_forward is not None and all(
            hasattr(feed_forward, name) for name in ("w1", "w2", "w3")
        ):
            sites.append(
                _site(
                    layer=layer_index,
                    family="ffn",
                    kind="ffn_out",
                    path=f"{prefix}.{layer_index}.feed_forward.w2",
                    module=feed_forward.w2,
                    stream_dim=stream_dim,
                )
            )

        sites.append(
            _site(
                layer=layer_index,
                family="block",
                kind="block_out",
                path=f"{prefix}.{layer_index}",
                module=layer,
                stream_dim=stream_dim,
            )
        )

    payload = [site.__dict__ for site in sites]
    return SemanticSiteRegistry(tuple(sites), sha256_json(payload))


def assert_lfm25_layout(registry: SemanticSiteRegistry) -> None:
    """Fail closed when the pinned LFM2.5 pilot architecture changes."""

    counts = {family: len(registry.by_family(family)) for family in ("liv", "gqa", "ffn", "block")}
    expected = {"liv": 10, "gqa": 6, "ffn": 16, "block": 16}
    if counts != expected:
        raise RuntimeError(f"unexpected LFM2.5 semantic layout: {counts}, expected {expected}")
