"""Architecture-aware semantic intervention-site discovery."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
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


def _nested_attr(root: object, path: str) -> object | None:
    value: object | None = root
    for part in path.split("."):
        value = getattr(value, part, None)
        if value is None:
            return None
    return value


def _model_layers(model: nn.Module) -> tuple[str, list[nn.Module]]:
    candidates = (
        "model.layers",
        "model.decoder.layers",
        "transformer.h",
        "gpt_neox.layers",
        "layers",
    )
    for prefix in candidates:
        layers = _nested_attr(model, prefix)
        if layers is not None and isinstance(layers, (nn.ModuleList, list, tuple)):
            return prefix, list(layers)
    raise ValueError("model does not expose a supported decoder-layer collection")


def _first_module(root: nn.Module, names: Sequence[str]) -> tuple[str, nn.Module] | None:
    for name in names:
        value = getattr(root, name, None)
        if isinstance(value, nn.Module):
            return name, value
    return None


def _configured_hidden_size(model: nn.Module) -> int:
    config = getattr(model, "config", None)
    text_config = getattr(config, "text_config", None)
    for source in (config, text_config):
        for name in ("hidden_size", "d_model", "n_embd"):
            value = getattr(source, name, None)
            if value is not None and int(value) > 0:
                return int(value)
    return 0


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
    configured_dim = _configured_hidden_size(model)
    sites: list[SemanticSite] = []
    for layer_index, layer in enumerate(layers):
        feed_forward_match = _first_module(layer, ("feed_forward", "mlp"))
        feed_forward_name, feed_forward = (
            feed_forward_match if feed_forward_match is not None else ("", None)
        )
        ffn_output_match = (
            _first_module(
                feed_forward,
                ("w2", "down_proj", "fc2", "dense_4h_to_h"),
            )
            if feed_forward is not None
            else None
        )
        stream_dim = configured_dim
        if ffn_output_match is not None:
            _ffn_output_name, ffn_output = ffn_output_match
            _ffn_in, ffn_out = _linear_shape(ffn_output)
            stream_dim = int(ffn_out or stream_dim)
        if stream_dim <= 0:
            raise ValueError(f"cannot infer stream dimension at layer {layer_index}")

        attention_match = _first_module(layer, ("self_attn", "attention", "attn"))
        attention_name, attention = (
            attention_match if attention_match is not None else ("", None)
        )
        attention_output_match = (
            _first_module(attention, ("out_proj", "o_proj", "dense"))
            if attention is not None
            else None
        )
        has_split_qkv = attention is not None and all(
            hasattr(attention, name) for name in ("q_proj", "k_proj", "v_proj")
        )
        if attention_output_match is not None and has_split_qkv:
            attention_output_name, attention_output = attention_output_match
            sites.append(
                _site(
                    layer=layer_index,
                    family="gqa",
                    kind="attention_out",
                    path=(
                        f"{prefix}.{layer_index}.{attention_name}."
                        f"{attention_output_name}"
                    ),
                    module=attention_output,
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

        if ffn_output_match is not None:
            ffn_output_name, ffn_output = ffn_output_match
            sites.append(
                _site(
                    layer=layer_index,
                    family="ffn",
                    kind="ffn_out",
                    path=(
                        f"{prefix}.{layer_index}.{feed_forward_name}."
                        f"{ffn_output_name}"
                    ),
                    module=ffn_output,
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
