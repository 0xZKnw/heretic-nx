"""Fail-closed structural discovery for model-agnostic intervention sites.

This module deliberately separates activation capture points from weight
storage.  A projection can expose a useful activation site without using the
usual ``[out, in]`` storage convention, and a routed expert bank can expose an
editable-looking tensor without providing a proven activation boundary.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from enum import Enum
from typing import Literal

import torch
from torch import Tensor, nn

from heretic_nx.hashing import sha256_json


ActivationRole = Literal[
    "attention_output",
    "ffn_output",
    "shared_ffn_output",
]
AttentionVariant = Literal["split", "fused", "mla", "shared_kv"]
WeightRole = Literal[
    "attention_output",
    "ffn_output",
    "shared_ffn_output",
    "routed_ffn_output",
]
DecisionSubject = Literal["decoder_stack", "activation_site", "weight_target", "candidate"]


class WeightLayout(str, Enum):
    """Logical storage order for a projection weight."""

    OI = "OI"
    IO = "IO"
    EOI = "EOI"


@dataclass(frozen=True)
class ActivationSite:
    """A module output that can be observed without assuming weight layout."""

    id: str
    layer: int
    role: ActivationRole
    module_path: str
    stream_dim: int
    attention_variant: AttentionVariant | None


@dataclass(frozen=True)
class WeightTarget:
    """A concrete projection parameter bound to a semantic role."""

    id: str
    layer: int
    role: WeightRole
    module_path: str
    parameter_path: str
    activation_site_id: str | None
    layout: WeightLayout
    shape: tuple[int, ...]
    input_dim: int
    output_dim: int
    expert_count: int | None
    dtype: str
    editable: bool


@dataclass(frozen=True)
class DiscoveryRecord:
    """One auditable inclusion, exclusion, or ambiguity decision."""

    subject: DecisionSubject
    code: str
    path: str
    layer: int | None
    object_id: str | None
    detail: str


@dataclass(frozen=True)
class StructuralDiscoveryReport:
    """Complete deterministic report for one decoder discovery pass."""

    decoder_stack_path: str | None
    stream_dim: int | None
    layer_count: int
    activation_sites: tuple[ActivationSite, ...]
    weight_targets: tuple[WeightTarget, ...]
    inclusions: tuple[DiscoveryRecord, ...]
    exclusions: tuple[DiscoveryRecord, ...]
    ambiguities: tuple[DiscoveryRecord, ...]
    structure_hash: str

    @property
    def is_unambiguous(self) -> bool:
        return not self.ambiguities

    @property
    def editable_targets(self) -> tuple[WeightTarget, ...]:
        if self.ambiguities:
            return ()
        return tuple(target for target in self.weight_targets if target.editable)

    @property
    def candidate_targets(self) -> tuple[WeightTarget, ...]:
        """Return described targets, including fail-closed audit candidates."""

        return self.weight_targets

    def targets_by_role(self, role: WeightRole) -> tuple[WeightTarget, ...]:
        return tuple(target for target in self.weight_targets if target.role == role)


class StructuralDiscoveryError(ValueError):
    """Raised when a structure cannot be selected without guessing."""

    def __init__(self, report: StructuralDiscoveryReport) -> None:
        self.report = report
        summary = "; ".join(
            f"{item.code} at {item.path}: {item.detail}"
            for item in report.ambiguities
        )
        super().__init__(f"ambiguous model structure: {summary}")


@dataclass(frozen=True)
class _Projection:
    module_path: str
    parameter_path: str
    module: nn.Module
    weight: Tensor
    layout: WeightLayout
    input_dim: int
    output_dim: int
    expert_count: int | None


_STACK_TERMINALS = frozenset({"layers", "blocks", "h"})
_ATTENTION_OUTPUT_NAMES = frozenset(
    {"o_proj", "out_proj", "output_proj", "c_proj", "dense", "wo", "proj"}
)
_FFN_OUTPUT_NAMES = frozenset(
    {
        "down_proj",
        "w2",
        "fc2",
        "fc_out",
        "c_proj",
        "dense_4h_to_h",
        "output_proj",
        "out_proj",
        "proj",
    }
)
_QUERY_NAMES = frozenset(
    {"q_proj", "query", "query_proj", "wq", "q_a_proj", "q_b_proj"}
)
_KEY_NAMES = frozenset({"k_proj", "key", "key_proj", "wk"})
_VALUE_NAMES = frozenset({"v_proj", "value", "value_proj", "wv"})
_FUSED_QKV_NAMES = frozenset(
    {"qkv_proj", "query_key_value", "c_attn", "wqkv", "w_pack"}
)
_MLA_NAMES = frozenset(
    {
        "q_a_proj",
        "q_b_proj",
        "kv_a_proj",
        "kv_a_proj_with_mqa",
        "kv_b_proj",
    }
)
_FFN_EXPANSION_NAMES = frozenset(
    {
        "up_proj",
        "gate_proj",
        "gate_up_proj",
        "w1",
        "w3",
        "fc1",
        "fc_in",
        "c_fc",
        "dense_h_to_4h",
    }
)
_ROUTED_COMPONENT_NAMES = frozenset(
    {"experts", "expert", "expert_bank", "routed_experts", "routed_expert"}
)
_SHARED_COMPONENT_NAMES = frozenset(
    {"shared_expert", "shared_experts", "shared_ffn", "shared_mlp"}
)
_NON_TEXT_MODALITY_TOKENS = frozenset(
    {"audio", "image", "speech", "visual", "vision"}
)
_TEXT_MODALITY_TOKENS = frozenset({"language", "text"})
_TRANSFORMERS_CONV1D = ("transformers.pytorch_utils", "Conv1D")
_TRUSTED_LINEAR_OVERRIDES = frozenset(
    {
        ("transformers.models.falcon.modeling_falcon", "FalconLinear"),
    }
)


def _leaf(path: str) -> str:
    return path.rsplit(".", 1)[-1].lower()


def _path_parts(path: str) -> frozenset[str]:
    return frozenset(part.lower() for part in path.split("."))


def _dtype_name(weight: Tensor) -> str:
    return str(weight.dtype).removeprefix("torch.")


def _decision(
    subject: DecisionSubject,
    code: str,
    path: str,
    *,
    layer: int | None = None,
    object_id: str | None = None,
    detail: str,
) -> DiscoveryRecord:
    return DiscoveryRecord(subject, code, path, layer, object_id, detail)


def _decoder_stack_candidates(model: nn.Module) -> tuple[tuple[str, nn.Module], ...]:
    aliases: dict[int, list[tuple[str, nn.Module]]] = {}
    for path, module in model.named_modules(remove_duplicate=False):
        if not path or _leaf(path) not in _STACK_TERMINALS:
            continue
        if not isinstance(module, (nn.ModuleList, nn.Sequential)) or len(module) == 0:
            continue
        if not all(isinstance(layer, nn.Module) for layer in module):
            continue
        aliases.setdefault(id(module), []).append((path, module))
    candidates = [
        min(paths, key=lambda item: item[0])
        for paths in aliases.values()
    ]
    return tuple(sorted(candidates, key=lambda item: item[0]))


def _non_text_stack_reason(path: str) -> str | None:
    tokens = frozenset(
        token
        for component in path.lower().split(".")
        for token in component.replace("-", "_").split("_")
        if token
    )
    non_text = sorted(tokens & _NON_TEXT_MODALITY_TOKENS)
    if non_text and not tokens & _TEXT_MODALITY_TOKENS:
        return f"stack path is explicitly non-textual: {non_text}"
    return None


def _configured_stream_dimensions(model: nn.Module) -> tuple[int, ...]:
    values: set[int] = set()
    pending: list[object] = [getattr(model, "config", None)]
    seen: set[int] = set()
    while pending:
        source = pending.pop()
        if source is None or id(source) in seen:
            continue
        seen.add(id(source))
        for name in ("hidden_size", "d_model", "n_embd", "model_dim"):
            value = getattr(source, name, None)
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                values.add(value)
        for name in ("text_config", "decoder", "language_config"):
            nested = getattr(source, name, None)
            if nested is not None:
                pending.append(nested)
    return tuple(sorted(values))


def _normalized_stream_dimensions(layers: nn.Module) -> tuple[int, ...]:
    values: set[int] = set()
    for layer in layers:
        for module in layer.modules():
            shape = getattr(module, "normalized_shape", None)
            if isinstance(shape, int) and shape > 0:
                values.add(shape)
            elif isinstance(shape, (tuple, list)) and len(shape) == 1:
                value = shape[0]
                if isinstance(value, int) and value > 0:
                    values.add(value)
    return tuple(sorted(values))


def _infer_stream_dim(model: nn.Module, layers: nn.Module) -> tuple[int | None, str | None]:
    configured = _configured_stream_dimensions(model)
    if len(configured) > 1:
        return None, f"conflicting configured residual widths: {list(configured)}"
    normalized = _normalized_stream_dimensions(layers)
    if len(normalized) > 1:
        return None, f"conflicting normalization widths: {list(normalized)}"
    if configured and normalized and configured[0] != normalized[0]:
        return None, (
            "configured and normalized residual widths disagree: "
            f"configured={configured[0]}, normalized={normalized[0]}"
        )
    if configured:
        return configured[0], None
    if normalized:
        return normalized[0], None
    return None, "no unique configured or normalized residual width"


def _declared_projection_layout(module: nn.Module) -> WeightLayout | None:
    """Return a layout only for module types with a proven Tensor contract."""

    module_type = type(module)
    if (
        "forward" in module.__dict__
        or "__call__" in module.__dict__
        or "_call_impl" in module.__dict__
        or module_type.__call__ is not nn.Module.__call__
        or module_type._call_impl is not nn.Module._call_impl
        or bool(module._forward_hooks)
        or bool(module._forward_pre_hooks)
    ):
        return None
    if isinstance(module, nn.Linear):
        identity = (module_type.__module__, module_type.__qualname__)
        if (
            module_type.forward is nn.Linear.forward
            or identity in _TRUSTED_LINEAR_OVERRIDES
        ):
            return WeightLayout.OI
    if (module_type.__module__, module_type.__qualname__) == _TRANSFORMERS_CONV1D:
        return WeightLayout.IO
    return None


def _projection_from_module(
    module_path: str,
    module: nn.Module,
    stream_dim: int,
) -> tuple[_Projection | None, str | None]:
    weight = getattr(module, "weight", None)
    if not isinstance(weight, nn.Parameter):
        return None, "module has no directly registered Parameter weight"
    if module._parameters.get("weight") is not weight:
        return None, "module weight is not its canonical directly registered parameter"
    if isinstance(weight, nn.parameter.UninitializedParameter):
        return None, "module weight is an uninitialized lazy parameter"
    try:
        shape = tuple(int(value) for value in weight.shape)
        weight_rank = weight.ndim
    except RuntimeError:
        return None, "module weight metadata is not initialized"
    if weight_rank == 2:
        declared_layout = _declared_projection_layout(module)
        if declared_layout is None:
            return None, (
                "two-dimensional module has no proven Tensor-output projection "
                "contract"
            )
        in_features = getattr(module, "in_features", None)
        out_features = getattr(module, "out_features", None)
        nf = getattr(module, "nf", None)
        nx = getattr(module, "nx", None)
        if declared_layout is WeightLayout.OI:
            if not isinstance(in_features, int) or not isinstance(out_features, int):
                return None, "OI projection lacks integer in_features/out_features"
            if shape != (out_features, in_features):
                return None, (
                    "declared in_features/out_features disagree with the weight shape "
                    f"{shape}"
                )
            layout = WeightLayout.OI
            input_dim, output_dim = in_features, out_features
        else:
            if not isinstance(nx, int) or not isinstance(nf, int):
                return None, "IO projection lacks integer nx/nf dimensions"
            if shape != (nx, nf):
                return None, f"declared nx/nf disagree with the weight shape {shape}"
            layout = WeightLayout.IO
            input_dim, output_dim = nx, nf
        return (
            _Projection(
                module_path=module_path,
                parameter_path=f"{module_path}.weight",
                module=module,
                weight=weight,
                layout=layout,
                input_dim=input_dim,
                output_dim=output_dim,
                expert_count=None,
            ),
            None,
        )
    if weight_rank == 3:
        if shape[1] == stream_dim and shape[2] == stream_dim:
            return None, (
                f"three-dimensional shape {shape} is ambiguous between EOI and EIO"
            )
        if shape[1] == stream_dim:
            return (
                _Projection(
                    module_path=module_path,
                    parameter_path=f"{module_path}.weight",
                    module=module,
                    weight=weight,
                    layout=WeightLayout.EOI,
                    input_dim=shape[2],
                    output_dim=shape[1],
                    expert_count=shape[0],
                ),
                None,
            )
        if shape[2] == stream_dim:
            return None, (
                f"three-dimensional shape {shape} looks EIO; only proven EOI storage is "
                "accepted"
            )
        return None, (
            f"three-dimensional shape {shape} does not emit residual width {stream_dim}"
        )
    return None, f"weight rank {weight_rank} is not a projection matrix or expert bank"


def _projection_from_direct_parameter(
    *,
    module_path: str,
    parameter_name: str,
    module: nn.Module,
    weight: Tensor,
    stream_dim: int,
) -> tuple[_Projection | None, str | None]:
    """Describe fused expert parameters that are not wrapped by a Linear module.

    A direct two-dimensional parameter has no independently hookable output
    boundary, so phase 1 refuses to manufacture an ``ActivationSite`` for it.
    The EOI bank is still structurally useful and is deliberately non-editable.
    """

    if isinstance(weight, nn.parameter.UninitializedParameter):
        return None, "direct weight is an uninitialized lazy parameter"
    try:
        shape = tuple(int(value) for value in weight.shape)
        weight_rank = weight.ndim
    except RuntimeError:
        return None, "direct weight metadata is not initialized"
    if weight_rank != 3:
        return None, (
            "direct projection parameters are accepted only for descriptive EOI "
            "expert banks; no activation boundary is proven"
        )
    if shape[1] == stream_dim and shape[2] == stream_dim:
        return None, (
            f"three-dimensional shape {shape} is ambiguous between EOI and EIO"
        )
    if shape[1] != stream_dim:
        if shape[2] == stream_dim:
            return None, (
                f"three-dimensional shape {shape} looks EIO; only proven EOI storage is "
                "accepted"
            )
        return None, (
            f"three-dimensional shape {shape} does not emit residual width {stream_dim}"
        )
    return (
        _Projection(
            module_path=module_path,
            parameter_path=f"{module_path}.{parameter_name}",
            module=module,
            weight=weight,
            layout=WeightLayout.EOI,
            input_dim=shape[2],
            output_dim=shape[1],
            expert_count=shape[0],
        ),
        None,
    )


def _descendant_leaf_names(module: nn.Module) -> frozenset[str]:
    names = {_leaf(path) for path, _child in module.named_modules() if path}
    names.update(_leaf(path) for path, _parameter in module.named_parameters())
    return frozenset(names)


def _contains_sequence_operator(module: nn.Module) -> bool:
    for descendant in module.modules():
        if isinstance(descendant, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            return True
        name = type(descendant).__name__.lower()
        if any(marker in name for marker in ("conv", "mamba", "recurrent", "ssm")):
            return True
    return False


def _direct_semantic_children(
    layer: nn.Module,
) -> tuple[tuple[str, nn.Module, str, bool], ...]:
    matches: list[tuple[str, nn.Module, str, bool]] = []
    direct_children = tuple(
        (name, child)
        for name, child in layer._modules.items()
        if isinstance(child, nn.Module)
    )
    for name, child in direct_children:
        leaves = _descendant_leaf_names(child)
        child_leaf = name.lower()
        has_output = bool(leaves & (_ATTENTION_OUTPUT_NAMES | _FFN_OUTPUT_NAMES))
        attention_evidence = bool(
            leaves & (_QUERY_NAMES | _KEY_NAMES | _VALUE_NAMES | _FUSED_QKV_NAMES | _MLA_NAMES)
        )
        ffn_evidence = bool(leaves & _FFN_EXPANSION_NAMES) or bool(
            leaves & (_ROUTED_COMPONENT_NAMES | _SHARED_COMPONENT_NAMES)
        ) or child_leaf in (_ROUTED_COMPONENT_NAMES | _SHARED_COMPONENT_NAMES)
        if has_output and attention_evidence and ffn_evidence:
            matches.append((name, child, "ambiguous", False))
        elif has_output and attention_evidence:
            matches.append((name, child, "attention", False))
        elif has_output and ffn_evidence:
            matches.append((name, child, "ffn", False))
        elif has_output and _contains_sequence_operator(child):
            matches.append((name, child, "operator", False))
        elif has_output:
            matches.append((name, child, "unclassified", False))

    # Some decoder blocks expose their projections directly instead of
    # wrapping the FFN or attention in a named submodule.  Direct-only scopes
    # avoid accidentally consuming projections from already-recognized nested
    # branches (for example an attention ``out_proj`` next to block-level fc2).
    direct_names = frozenset(name.lower() for name, _child in direct_children)
    direct_attention = bool(
        direct_names
        & (_QUERY_NAMES | _KEY_NAMES | _VALUE_NAMES | _FUSED_QKV_NAMES | _MLA_NAMES)
    ) and bool(direct_names & _ATTENTION_OUTPUT_NAMES)
    direct_ffn = bool(direct_names & _FFN_EXPANSION_NAMES) and bool(
        direct_names & _FFN_OUTPUT_NAMES
    )
    if direct_attention:
        matches.append(("", layer, "attention", True))
    if direct_ffn:
        matches.append(("", layer, "ffn", True))
    return tuple(matches)


def _named_projections(
    module: nn.Module,
    names: frozenset[str],
    stream_dim: int,
) -> tuple[_Projection, ...]:
    projections: list[_Projection] = []
    for path, descendant in module.named_modules():
        if not path or _leaf(path) not in names:
            continue
        projection, _error = _projection_from_module(path, descendant, stream_dim)
        if projection is not None:
            projections.append(projection)
    return tuple(projections)


def _attention_variant(
    module: nn.Module, stream_dim: int
) -> AttentionVariant | None:
    direct_queries = _named_projections(module, frozenset({"q_proj"}), stream_dim)
    query_a = _named_projections(module, frozenset({"q_a_proj"}), stream_dim)
    query_b = _named_projections(module, frozenset({"q_b_proj"}), stream_dim)
    kv_a = _named_projections(
        module,
        frozenset({"kv_a_proj", "kv_a_proj_with_mqa"}),
        stream_dim,
    )
    kv_b = _named_projections(module, frozenset({"kv_b_proj"}), stream_dim)
    coherent_direct_query = (
        len(direct_queries) == 1 and direct_queries[0].input_dim == stream_dim
    )
    coherent_lora_query = (
        len(query_a) == len(query_b) == 1
        and query_a[0].input_dim == stream_dim
        and query_b[0].input_dim == query_a[0].output_dim
    )
    coherent_mla_kv = (
        len(kv_a) == len(kv_b) == 1
        and kv_a[0].input_dim == stream_dim
        and 0 < kv_b[0].input_dim <= kv_a[0].output_dim
    )
    if (coherent_direct_query or coherent_lora_query) and coherent_mla_kv:
        return "mla"
    fused = _named_projections(module, _FUSED_QKV_NAMES, stream_dim)
    if (
        len(fused) == 1
        and fused[0].input_dim == stream_dim
        and fused[0].output_dim > stream_dim
    ):
        return "fused"
    queries = _named_projections(
        module,
        frozenset({"q_proj", "query", "query_proj", "wq"}),
        stream_dim,
    )
    keys = _named_projections(module, _KEY_NAMES, stream_dim)
    values = _named_projections(module, _VALUE_NAMES, stream_dim)
    if (
        len(queries) == len(keys) == len(values) == 1
        and queries[0].input_dim == stream_dim
        and keys[0].input_dim == stream_dim
        and values[0].input_dim == stream_dim
    ):
        return "split"
    return None


def _ffn_structure_is_coherent(
    module: nn.Module,
    *,
    child_path: str,
    direct_only: bool,
    stream_dim: int,
    outputs: tuple[_Projection, ...],
) -> bool:
    """Require a proven expansion path before labeling a dense FFN output."""

    if _path_parts(child_path) & _ROUTED_COMPONENT_NAMES:
        return True
    leaves = _descendant_leaf_names(module)
    if leaves & _ROUTED_COMPONENT_NAMES:
        return True
    expansions = tuple(
        projection
        for projection in _named_projections(
            module,
            _FFN_EXPANSION_NAMES,
            stream_dim,
        )
        if projection.input_dim == stream_dim
        and projection.output_dim > 0
        and projection.output_dim != stream_dim
        and (not direct_only or projection.module_path.count(".") == 0)
    )
    return any(
        expansion.output_dim == output.input_dim
        or expansion.output_dim == 2 * output.input_dim
        for expansion in expansions
        for output in outputs
    )


def _output_candidates(
    *,
    base_path: str,
    module: nn.Module,
    names: frozenset[str],
    stream_dim: int,
    direct_only: bool = False,
) -> tuple[tuple[_Projection, ...], tuple[tuple[str, str], ...]]:
    projections: list[_Projection] = []
    rejected: list[tuple[str, str]] = []
    seen_parameters: set[str] = set()
    for relative_path, descendant in module.named_modules():
        if direct_only and relative_path.count(".") > 0:
            continue
        owner_path = base_path if not relative_path else f"{base_path}.{relative_path}"
        if relative_path and _leaf(relative_path) in names:
            projection, error = _projection_from_module(owner_path, descendant, stream_dim)
            if projection is None:
                rejected.append((owner_path, error or "not a supported projection"))
            elif projection.layout is WeightLayout.EOI and not (
                _path_parts(owner_path) & _ROUTED_COMPONENT_NAMES
            ):
                rejected.append(
                    (
                        owner_path,
                        "three-dimensional output storage is accepted only inside an "
                        "explicit routed-expert path",
                    )
                )
            elif projection.output_dim != stream_dim:
                rejected.append(
                    (
                        owner_path,
                        f"projection emits {projection.output_dim}, not residual width {stream_dim}",
                    )
                )
            else:
                projections.append(projection)
                seen_parameters.add(projection.parameter_path)
        for parameter_name, parameter in descendant.named_parameters(recurse=False):
            if parameter_name == "weight" or _leaf(parameter_name) not in names:
                continue
            parameter_path = f"{owner_path}.{parameter_name}"
            if parameter_path in seen_parameters:
                continue
            projection, error = _projection_from_direct_parameter(
                module_path=owner_path,
                parameter_name=parameter_name,
                module=descendant,
                weight=parameter,
                stream_dim=stream_dim,
            )
            if projection is None:
                rejected.append((parameter_path, error or "not a supported projection"))
            elif not (_path_parts(owner_path) & _ROUTED_COMPONENT_NAMES):
                rejected.append(
                    (
                        parameter_path,
                        "direct EOI storage is accepted only inside an explicit "
                        "routed-expert path",
                    )
                )
            else:
                projections.append(projection)
                seen_parameters.add(projection.parameter_path)
    return tuple(projections), tuple(rejected)


def _activation_id(layer: int, role: ActivationRole, module_path: str) -> str:
    return f"activation::layer={layer}::role={role}::path={module_path}"


def _weight_id(layer: int, role: WeightRole, parameter_path: str) -> str:
    return f"weight::layer={layer}::role={role}::path={parameter_path}"


def _make_activation(
    *,
    layer: int,
    role: ActivationRole,
    projection: _Projection,
    stream_dim: int,
    attention_variant: AttentionVariant | None = None,
) -> ActivationSite:
    return ActivationSite(
        id=_activation_id(layer, role, projection.module_path),
        layer=layer,
        role=role,
        module_path=projection.module_path,
        stream_dim=stream_dim,
        attention_variant=attention_variant,
    )


def _make_target(
    *,
    layer: int,
    role: WeightRole,
    projection: _Projection,
    activation_site_id: str | None,
    editable: bool,
) -> WeightTarget:
    return WeightTarget(
        id=_weight_id(layer, role, projection.parameter_path),
        layer=layer,
        role=role,
        module_path=projection.module_path,
        parameter_path=projection.parameter_path,
        activation_site_id=activation_site_id,
        layout=projection.layout,
        shape=tuple(int(value) for value in projection.weight.shape),
        input_dim=projection.input_dim,
        output_dim=projection.output_dim,
        expert_count=projection.expert_count,
        dtype=_dtype_name(projection.weight),
        editable=editable,
    )


def _projection_editability_error(projection: _Projection) -> str | None:
    weight = projection.weight
    if projection.layout is WeightLayout.EOI:
        return "routed expert banks require routing-aware evidence"
    if weight.layout is not torch.strided:
        return f"tensor layout {weight.layout} is not an editable strided matrix"
    if not weight.is_floating_point():
        return f"dtype {_dtype_name(weight)} is not a floating edit workspace"
    return None


def _storage_alias_key(parameter: Tensor) -> tuple[object, ...]:
    """Return a process-local storage key that is never serialized."""

    try:
        storage = parameter.untyped_storage()
        return ("storage", str(parameter.device), int(storage._cdata))
    except (AttributeError, NotImplementedError, RuntimeError, ValueError):
        return ("parameter", id(parameter))


def _parameter_schema(
    stack_path: str | None,
    layers: nn.Module | None,
    stream_dim: int | None,
) -> tuple[dict[str, object], ...]:
    if stack_path is None or layers is None:
        return ()
    parameters = tuple(layers.named_parameters(remove_duplicate=False))
    alias_paths: dict[tuple[object, ...], list[str]] = {}
    for relative_path, parameter in parameters:
        alias_paths.setdefault(_storage_alias_key(parameter), []).append(
            f"{stack_path}.{relative_path}"
        )
    schemas: list[dict[str, object]] = []
    for relative_path, parameter in parameters:
        path = f"{stack_path}.{relative_path}"
        try:
            shape: tuple[int, ...] | None = tuple(
                int(value) for value in parameter.shape
            )
            rank: int | None = parameter.ndim
        except RuntimeError:
            shape = None
            rank = None
        layout: str | None = None
        expert_count: int | None = None
        if rank == 3 and shape is not None and _path_parts(path) & _ROUTED_COMPONENT_NAMES:
            expert_count = shape[0]
            if (
                stream_dim is not None
                and shape[1] == stream_dim
                and shape[2] != stream_dim
            ):
                layout = WeightLayout.EOI.value
            else:
                layout = "unknown-3d"
        schemas.append(
            {
                "path": path,
                "shape": shape,
                "rank": rank,
                "dtype": _dtype_name(parameter),
                "layout": layout,
                "expert_count": expert_count,
                "shared_storage_paths": tuple(
                    sorted(alias_paths[_storage_alias_key(parameter)])
                ),
            }
        )
    return tuple(sorted(schemas, key=lambda item: str(item["path"])))


def _shared_target_ambiguities(
    model: nn.Module,
    targets: list[WeightTarget],
) -> tuple[DiscoveryRecord, ...]:
    parameter_paths: dict[tuple[object, ...], set[str]] = {}
    for path, parameter in model.named_parameters(remove_duplicate=False):
        parameter_paths.setdefault(_storage_alias_key(parameter), set()).add(path)
    records: list[DiscoveryRecord] = []
    for target in targets:
        try:
            parameter = model.get_parameter(target.parameter_path)
        except (AttributeError, KeyError, TypeError):
            records.append(
                _decision(
                    "weight_target",
                    "weight_target_path_invalid",
                    target.parameter_path,
                    layer=target.layer,
                    object_id=target.id,
                    detail="target path does not resolve to a canonical Parameter",
                )
            )
            continue
        paths = sorted(parameter_paths.get(_storage_alias_key(parameter), set()))
        if len(paths) < 2:
            continue
        records.append(
            _decision(
                "weight_target",
                "weight_target_storage_shared",
                target.parameter_path,
                layer=target.layer,
                object_id=target.id,
                detail=(
                    "target storage is reachable through multiple parameter paths: "
                    f"{paths}"
                ),
            )
        )
    return tuple(records)


def _semantic_role_ambiguities(
    activation_sites: list[ActivationSite],
    weight_targets: list[WeightTarget],
) -> tuple[DiscoveryRecord, ...]:
    groups: dict[tuple[int, str], list[str]] = {}
    for site in activation_sites:
        groups.setdefault((site.layer, site.role), []).append(site.module_path)
    for target in weight_targets:
        if target.activation_site_id is None:
            groups.setdefault((target.layer, target.role), []).append(
                target.parameter_path
            )
    records: list[DiscoveryRecord] = []
    for (layer, role), paths in sorted(groups.items()):
        unique_paths = sorted(set(paths))
        if len(unique_paths) < 2:
            continue
        records.append(
            _decision(
                "candidate",
                f"{role}_not_unique",
                unique_paths[0],
                layer=layer,
                detail=f"role {role!r} has multiple candidates: {unique_paths}",
            )
        )
    return tuple(records)


def _finalize_report(
    *,
    decoder_stack_path: str | None,
    stream_dim: int | None,
    layer_count: int,
    activation_sites: list[ActivationSite],
    weight_targets: list[WeightTarget],
    inclusions: list[DiscoveryRecord],
    exclusions: list[DiscoveryRecord],
    ambiguities: list[DiscoveryRecord],
    parameter_schema: tuple[dict[str, object], ...],
) -> StructuralDiscoveryReport:
    activation_sites.sort(key=lambda item: item.id)
    weight_targets.sort(key=lambda item: item.id)
    inclusions.sort(key=lambda item: (item.path, item.code, item.object_id or ""))
    exclusions.sort(key=lambda item: (item.path, item.code, item.object_id or ""))
    ambiguities.sort(key=lambda item: (item.path, item.code, item.object_id or ""))

    activation_ids = [item.id for item in activation_sites]
    target_ids = [item.id for item in weight_targets]
    duplicate_ids = sorted(
        identifier
        for identifier in set(activation_ids + target_ids)
        if (activation_ids + target_ids).count(identifier) > 1
    )
    for identifier in duplicate_ids:
        ambiguities.append(
            _decision(
                "candidate",
                "duplicate_identifier",
                identifier,
                object_id=identifier,
                detail="structural IDs must be collision-free",
            )
        )
    ambiguities.sort(key=lambda item: (item.path, item.code, item.object_id or ""))

    # ``inspect_*`` is also a public API.  Never expose an effectively editable
    # target from a report that says the surrounding structure is ambiguous.
    if ambiguities:
        weight_targets[:] = [replace(item, editable=False) for item in weight_targets]

    payload = {
        "schema": "heretic-nx-structural-frontend-v2",
        "decoder_stack_path": decoder_stack_path,
        "stream_dim": stream_dim,
        "layer_count": layer_count,
        "parameters": parameter_schema,
        "activation_sites": [asdict(item) for item in activation_sites],
        "weight_targets": [
            {
                **asdict(item),
                "layout": item.layout.value,
            }
            for item in weight_targets
        ],
        "inclusions": [asdict(item) for item in inclusions],
        "exclusions": [asdict(item) for item in exclusions],
        "ambiguities": [asdict(item) for item in ambiguities],
    }
    return StructuralDiscoveryReport(
        decoder_stack_path=decoder_stack_path,
        stream_dim=stream_dim,
        layer_count=layer_count,
        activation_sites=tuple(activation_sites),
        weight_targets=tuple(weight_targets),
        inclusions=tuple(inclusions),
        exclusions=tuple(exclusions),
        ambiguities=tuple(ambiguities),
        structure_hash=sha256_json(payload),
    )


def inspect_structural_frontend(model: nn.Module) -> StructuralDiscoveryReport:
    """Inspect a decoder and return every structural decision without guessing."""

    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    inclusions: list[DiscoveryRecord] = []
    exclusions: list[DiscoveryRecord] = []
    ambiguities: list[DiscoveryRecord] = []
    activation_sites: list[ActivationSite] = []
    weight_targets: list[WeightTarget] = []

    discovered_stacks = _decoder_stack_candidates(model)
    stacks: list[tuple[str, nn.Module]] = []
    rejected_stack_details: list[str] = []
    for candidate_path, candidate_layers in discovered_stacks:
        reason = _non_text_stack_reason(candidate_path)
        if reason is None:
            stacks.append((candidate_path, candidate_layers))
            continue
        rejected_stack_details.append(f"{candidate_path}: {reason}")
        exclusions.append(
            _decision(
                "decoder_stack",
                "non_text_decoder_stack_excluded",
                candidate_path,
                detail=reason,
            )
        )
    if len(stacks) != 1:
        detail = (
            (
                "no text decoder collection remained; rejected stacks: "
                f"{sorted(rejected_stack_details)}"
                if rejected_stack_details
                else "no non-empty layer/block collection was found"
            )
            if not stacks
            else (
                "multiple text decoder collections were found: "
                f"{sorted(path for path, _ in stacks)}"
            )
        )
        ambiguities.append(
            _decision(
                "decoder_stack",
                "decoder_stack_not_unique",
                "<model>",
                detail=detail,
            )
        )
        return _finalize_report(
            decoder_stack_path=None,
            stream_dim=None,
            layer_count=0,
            activation_sites=activation_sites,
            weight_targets=weight_targets,
            inclusions=inclusions,
            exclusions=exclusions,
            ambiguities=ambiguities,
            parameter_schema=(),
        )

    stack_path, layers = stacks[0]
    inclusions.append(
        _decision(
            "decoder_stack",
            "decoder_stack_included",
            stack_path,
            detail=f"selected one structural collection with {len(layers)} layers",
        )
    )
    stream_dim, dimension_error = _infer_stream_dim(model, layers)
    if stream_dim is None:
        ambiguities.append(
            _decision(
                "decoder_stack",
                "stream_dimension_not_unique",
                stack_path,
                detail=dimension_error or "cannot infer residual width",
            )
        )
        return _finalize_report(
            decoder_stack_path=stack_path,
            stream_dim=None,
            layer_count=len(layers),
            activation_sites=activation_sites,
            weight_targets=weight_targets,
            inclusions=inclusions,
            exclusions=exclusions,
            ambiguities=ambiguities,
            parameter_schema=_parameter_schema(stack_path, layers, None),
        )

    # ``named_children`` removes duplicate modules.  The raw registration map
    # preserves both tied layer slots and non-numeric Sequential keys.
    layer_entries = tuple(
        (key, layer)
        for key, layer in layers._modules.items()
        if isinstance(layer, nn.Module)
    )
    for layer_index, (layer_key, layer) in enumerate(layer_entries):
        layer_path = f"{stack_path}.{layer_key}"
        semantic_children = _direct_semantic_children(layer)
        if not semantic_children:
            ambiguities.append(
                _decision(
                    "candidate",
                    "layer_has_no_supported_output",
                    layer_path,
                    layer=layer_index,
                    detail=(
                        "no direct child exposes a structurally recognizable attention or "
                        "feed-forward residual output"
                    ),
                )
            )
            continue

        for child_name, child, family, direct_only in semantic_children:
            child_path = f"{layer_path}.{child_name}" if child_name else layer_path
            if family == "ambiguous":
                ambiguities.append(
                    _decision(
                        "candidate",
                        "semantic_family_ambiguous",
                        child_path,
                        layer=layer_index,
                        detail="child exposes both attention and feed-forward capabilities",
                    )
                )
                continue

            if family == "unclassified":
                projections, rejected = _output_candidates(
                    base_path=child_path,
                    module=child,
                    names=_ATTENTION_OUTPUT_NAMES | _FFN_OUTPUT_NAMES,
                    stream_dim=stream_dim,
                    direct_only=direct_only,
                )
                for path, detail in rejected:
                    exclusions.append(
                        _decision(
                            "candidate",
                            "unclassified_output_candidate_excluded",
                            path,
                            layer=layer_index,
                            detail=detail,
                        )
                    )
                proven_non_residual = bool(rejected) and all(
                    "not residual width" in detail for _path, detail in rejected
                )
                if projections or not proven_non_residual:
                    ambiguities.append(
                        _decision(
                            "candidate",
                            "residual_branch_unclassified",
                            child_path,
                            layer=layer_index,
                            detail=(
                                "output-like residual branch has neither a coherent "
                                "attention nor feed-forward structure"
                            ),
                        )
                    )
                continue

            if family == "operator":
                exclusions.append(
                    _decision(
                        "candidate",
                        "sequence_operator_deferred",
                        child_path,
                        layer=layer_index,
                        detail=(
                            "convolutional, recurrent, or state-space residual operators "
                            "need an explicit activation contract"
                        ),
                    )
                )
                ambiguities.append(
                    _decision(
                        "candidate",
                        "sequence_operator_unsupported",
                        child_path,
                        layer=layer_index,
                        detail=(
                            "refusing to label a sequence operator as attention or FFN"
                        ),
                    )
                )
                continue

            if family == "attention":
                variant = _attention_variant(child, stream_dim)
                projections, rejected = _output_candidates(
                    base_path=child_path,
                    module=child,
                    names=_ATTENTION_OUTPUT_NAMES,
                    stream_dim=stream_dim,
                    direct_only=direct_only,
                )
                for path, detail in rejected:
                    exclusions.append(
                        _decision(
                            "candidate",
                            "attention_candidate_excluded",
                            path,
                            layer=layer_index,
                            detail=detail,
                        )
                    )
                if variant is None or len(projections) != 1:
                    ambiguities.append(
                        _decision(
                            "candidate",
                            "attention_output_not_unique",
                            child_path,
                            layer=layer_index,
                            detail=(
                                f"variant={variant!r}, residual-width output candidates="
                                f"{sorted(item.module_path for item in projections)}"
                            ),
                        )
                    )
                    continue
                projection = projections[0]
                editability_error = _projection_editability_error(projection)
                site = _make_activation(
                    layer=layer_index,
                    role="attention_output",
                    projection=projection,
                    stream_dim=stream_dim,
                    attention_variant=variant,
                )
                target = _make_target(
                    layer=layer_index,
                    role="attention_output",
                    projection=projection,
                    activation_site_id=site.id,
                    editable=editability_error is None,
                )
                activation_sites.append(site)
                weight_targets.append(target)
                inclusions.extend(
                    (
                        _decision(
                            "activation_site",
                            "attention_output_included",
                            site.module_path,
                            layer=layer_index,
                            object_id=site.id,
                            detail=f"recognized {variant} attention residual output",
                        ),
                        _decision(
                            "weight_target",
                            "attention_weight_included",
                            target.parameter_path,
                            layer=layer_index,
                            object_id=target.id,
                            detail=f"recognized {target.layout.value} residual-width projection",
                        ),
                    )
                )
                if editability_error is not None:
                    exclusions.append(
                        _decision(
                            "weight_target",
                            "attention_weight_edit_disabled",
                            target.parameter_path,
                            layer=layer_index,
                            object_id=target.id,
                            detail=editability_error,
                        )
                    )
                continue

            projections, rejected = _output_candidates(
                base_path=child_path,
                module=child,
                names=_FFN_OUTPUT_NAMES,
                stream_dim=stream_dim,
                direct_only=direct_only,
            )
            for path, detail in rejected:
                exclusions.append(
                    _decision(
                        "candidate",
                        "ffn_candidate_excluded",
                        path,
                        layer=layer_index,
                        detail=detail,
                    )
                )
            if projections and not _ffn_structure_is_coherent(
                child,
                child_path=child_path,
                direct_only=direct_only,
                stream_dim=stream_dim,
                outputs=projections,
            ):
                ambiguities.append(
                    _decision(
                        "candidate",
                        "ffn_structure_incoherent",
                        child_path,
                        layer=layer_index,
                        detail=(
                            "no proven residual-width input expansion feeds the "
                            "candidate FFN output"
                        ),
                    )
                )
                continue
            dense: list[_Projection] = []
            shared: list[_Projection] = []
            routed: list[_Projection] = []
            for projection in projections:
                relative = projection.module_path.removeprefix(f"{layer_path}.")
                parts = _path_parts(relative)
                if projection.layout is WeightLayout.EOI or parts & _ROUTED_COMPONENT_NAMES:
                    routed.append(projection)
                elif parts & _SHARED_COMPONENT_NAMES:
                    shared.append(projection)
                else:
                    dense.append(projection)

            if len(dense) > 1 or len(shared) > 1 or len(routed) > 1:
                ambiguities.append(
                    _decision(
                        "candidate",
                        "ffn_output_not_unique",
                        child_path,
                        layer=layer_index,
                        detail=(
                            "residual-width candidates by role: "
                            f"dense={sorted(item.module_path for item in dense)}, "
                            f"shared={sorted(item.module_path for item in shared)}, "
                            f"routed={sorted(item.module_path for item in routed)}"
                        ),
                    )
                )
                continue
            if dense and (shared or routed):
                ambiguities.append(
                    _decision(
                        "candidate",
                        "ffn_dense_moe_roles_overlap",
                        child_path,
                        layer=layer_index,
                        detail="a feed-forward child cannot be both dense and shared/routed",
                    )
                )
                continue
            if not (dense or shared or routed):
                ambiguities.append(
                    _decision(
                        "candidate",
                        "ffn_output_not_found",
                        child_path,
                        layer=layer_index,
                        detail="no supported residual-width feed-forward projection was found",
                    )
                )
                continue

            for role, projection in (
                ("ffn_output", dense[0] if dense else None),
                ("shared_ffn_output", shared[0] if shared else None),
            ):
                if projection is None:
                    continue
                editability_error = _projection_editability_error(projection)
                typed_role: ActivationRole = role  # type: ignore[assignment]
                site = _make_activation(
                    layer=layer_index,
                    role=typed_role,
                    projection=projection,
                    stream_dim=stream_dim,
                )
                target = _make_target(
                    layer=layer_index,
                    role=typed_role,
                    projection=projection,
                    activation_site_id=site.id,
                    editable=editability_error is None,
                )
                if editability_error is not None:
                    exclusions.append(
                        _decision(
                            "weight_target",
                            f"{role}_weight_edit_disabled",
                            target.parameter_path,
                            layer=layer_index,
                            object_id=target.id,
                            detail=editability_error,
                        )
                    )
                activation_sites.append(site)
                weight_targets.append(target)
                inclusions.extend(
                    (
                        _decision(
                            "activation_site",
                            f"{role}_included",
                            site.module_path,
                            layer=layer_index,
                            object_id=site.id,
                            detail="recognized a residual-width feed-forward output",
                        ),
                        _decision(
                            "weight_target",
                            f"{role}_weight_included",
                            target.parameter_path,
                            layer=layer_index,
                            object_id=target.id,
                            detail=f"recognized {target.layout.value} projection storage",
                        ),
                    )
                )

            if routed:
                projection = routed[0]
                if projection.layout is not WeightLayout.EOI:
                    exclusions.append(
                        _decision(
                            "candidate",
                            "per_expert_matrix_edit_deferred",
                            projection.parameter_path,
                            layer=layer_index,
                            detail=(
                                "routed per-expert matrices need routing-aware causal evidence "
                                "before expert-specific editing"
                            ),
                        )
                    )
                else:
                    target = _make_target(
                        layer=layer_index,
                        role="routed_ffn_output",
                        projection=projection,
                        activation_site_id=None,
                        editable=False,
                    )
                    weight_targets.append(target)
                    inclusions.append(
                        _decision(
                            "weight_target",
                            "routed_ffn_bank_described",
                            target.parameter_path,
                            layer=layer_index,
                            object_id=target.id,
                            detail=(
                                f"described EOI bank with {target.expert_count} experts; "
                                "editing remains disabled"
                            ),
                        )
                    )
                    exclusions.append(
                        _decision(
                            "weight_target",
                            "routed_ffn_edit_deferred",
                            target.parameter_path,
                            layer=layer_index,
                            object_id=target.id,
                            detail=(
                                "no expert-specific edit is authorized without routing-aware "
                                "causal evidence"
                            ),
                        )
                    )

    layers_with_sites = {site.layer for site in activation_sites}
    for layer_index, (layer_key, _layer) in enumerate(layer_entries):
        if layer_index in layers_with_sites:
            continue
        ambiguities.append(
            _decision(
                "activation_site",
                "layer_has_no_editable_output",
                f"{stack_path}.{layer_key}",
                layer=layer_index,
                detail="no proven residual-stream activation boundary was discovered",
            )
        )

    ambiguities.extend(_semantic_role_ambiguities(activation_sites, weight_targets))
    ambiguities.extend(_shared_target_ambiguities(model, weight_targets))
    return _finalize_report(
        decoder_stack_path=stack_path,
        stream_dim=stream_dim,
        layer_count=len(layers),
        activation_sites=activation_sites,
        weight_targets=weight_targets,
        inclusions=inclusions,
        exclusions=exclusions,
        ambiguities=ambiguities,
        parameter_schema=_parameter_schema(stack_path, layers, stream_dim),
    )


def discover_structural_frontend(model: nn.Module) -> StructuralDiscoveryReport:
    """Return a structural registry, failing closed on every ambiguity."""

    report = inspect_structural_frontend(model)
    if report.ambiguities:
        raise StructuralDiscoveryError(report)
    return report


__all__ = [
    "ActivationSite",
    "DiscoveryRecord",
    "StructuralDiscoveryError",
    "StructuralDiscoveryReport",
    "WeightLayout",
    "WeightTarget",
    "discover_structural_frontend",
    "inspect_structural_frontend",
]
