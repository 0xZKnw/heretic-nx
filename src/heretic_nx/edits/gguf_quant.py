"""Transactional same-type edits for mixed-quantization GGUF files."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from safetensors.numpy import load as load_safetensors

from heretic_nx.hashing import canonical_json, sha256_file

from .gguf_codecs import GGUFQuantizationCodecRegistry, QUANT_LAYOUTS
from .gguf_q8 import GGUFQ8AblationPlan, _copy_source, _gguf_api


SHA256_PATTERN = r"^[0-9a-f]{64}$"
EditableQuantization = Literal[
    "Q2_K",
    "Q3_K",
    "Q4_K",
    "Q5_K",
    "Q6_K",
    "Q4_0",
    "Q4_1",
    "Q5_0",
    "Q5_1",
    "Q8_0",
]
GGUFArithmeticMode = Literal["legacy-plan-v2", "chunk-stable-v1"]

# Projection reductions are grouped by global row number, never by the plan's
# streaming chunk size.  This keeps payloads reproducible when operators choose
# a smaller chunk for memory or a larger chunk for throughput.
_PROJECTION_REDUCTION_ROWS = 128


class GGUFQuantizedTensorEdit(BaseModel):
    """One type-bound low-rank edit with quantization-aware guardrails."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    tensor_name: str = Field(min_length=1)
    expected_quantization: EditableQuantization
    a_key: str = Field(min_length=1)
    b_key: str | None = None
    right_key: str | None = None
    strength: float = Field(ge=0)
    preserve_row_norms: bool = True
    preserve_original_blocks: bool = True
    quantization_multipliers: tuple[float, ...] = Field(
        default=(1.0,), min_length=1, max_length=8
    )
    minimum_block_improvement: float = Field(default=0.0, ge=0.0, lt=1.0)
    require_payload_change: bool = True
    minimum_delta_cosine: float | None = Field(default=None, ge=-1.0, le=1.0)
    maximum_delta_relative_error: float | None = Field(default=None, ge=0.0)
    maximum_row_norm_relative_error: float | None = Field(default=None, ge=0.0)

    @model_validator(mode="after")
    def validate_edit(self) -> "GGUFQuantizedTensorEdit":
        if self.right_key is not None and self.b_key is not None:
            raise ValueError("right_key and b_key are mutually exclusive")
        if self.right_key is not None and self.preserve_row_norms:
            raise ValueError("direct right-factor edits cannot preserve row norms")
        if any(not math.isfinite(value) or value <= 0 for value in self.quantization_multipliers):
            raise ValueError("quantization multipliers must be finite and positive")
        if len(set(self.quantization_multipliers)) != len(self.quantization_multipliers):
            raise ValueError("quantization multipliers must be unique")
        if self.strength == 0 and (
            self.require_payload_change
            or self.minimum_delta_cosine is not None
            or self.maximum_delta_relative_error is not None
        ):
            raise ValueError(
                "zero-strength edits must disable payload and realized-delta gates"
            )
        return self

    @property
    def resolved_b_key(self) -> str:
        return self.b_key or self.a_key


class GGUFQuantizedAblationPlan(BaseModel):
    """Hash-bound plan for same-type edits in a mixed-quantization GGUF."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[
        "gguf-static-ablation-v2",
        "gguf-static-ablation-v3",
    ] = "gguf-static-ablation-v3"
    source_sha256: str = Field(pattern=SHA256_PATTERN)
    tensor_artifact_sha256: str = Field(pattern=SHA256_PATTERN)
    edits: tuple[GGUFQuantizedTensorEdit, ...] = Field(min_length=1)
    row_chunk_size: int = Field(default=128, ge=1, le=4096)
    verify_untouched_bytes: bool = True

    @model_validator(mode="after")
    def validate_unique_targets(self) -> "GGUFQuantizedAblationPlan":
        targets = [edit.tensor_name for edit in self.edits]
        if len(targets) != len(set(targets)):
            raise ValueError("each GGUF tensor may be edited at most once")
        return self

    @classmethod
    def read(cls, path: str | Path) -> "GGUFQuantizedAblationPlan":
        return cls.model_validate_json(Path(path).read_bytes())

    def write(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(canonical_json(self.model_dump()) + b"\n")


@dataclass(frozen=True)
class _ResolvedEdit:
    tensor_name: str
    expected_quantization: str
    a_key: str
    b_key: str | None
    right_key: str | None
    strength: float
    preserve_row_norms: bool
    preserve_original_blocks: bool
    quantization_multipliers: tuple[float, ...]
    minimum_block_improvement: float
    require_payload_change: bool
    minimum_delta_cosine: float | None
    maximum_delta_relative_error: float | None
    maximum_row_norm_relative_error: float | None

    @property
    def resolved_b_key(self) -> str:
        return self.b_key or self.a_key


def _sum_square_float64(value: np.ndarray) -> float:
    return float(
        np.sum(np.square(value, dtype=np.float64), dtype=np.float64)
    )


@dataclass
class _ErrorAccumulator:
    elements: int = 0
    quantization_sse: float = 0.0
    target_sum_square: float = 0.0
    delta_target_sum_square: float = 0.0
    delta_realized_sum_square: float = 0.0
    delta_error_sum_square: float = 0.0
    delta_dot: float = 0.0
    maximum_absolute_error: float = 0.0
    row_norm_relative_sum_square: float = 0.0
    row_norm_rows: int = 0
    maximum_row_norm_relative_error: float = 0.0
    total_blocks: int = 0
    changed_blocks: int = 0
    changed_bytes: int = 0
    tracked_chunk_array_bytes_lower_bound: int = 0
    choices: Counter[str] = field(default_factory=Counter)

    def update(
        self,
        *,
        base: np.ndarray,
        target: np.ndarray,
        realized: np.ndarray,
        original_raw: np.ndarray,
        selected_raw: np.ndarray,
        choice_indices: np.ndarray,
        multipliers: tuple[float, ...],
        type_size: int,
        workspace_bytes: int,
    ) -> None:
        error = realized - target
        target_delta = target - base
        realized_delta = realized - base
        delta_error = realized_delta - target_delta
        self.elements += target.size
        self.quantization_sse += _sum_square_float64(error)
        self.target_sum_square += _sum_square_float64(target)
        self.delta_target_sum_square += _sum_square_float64(target_delta)
        self.delta_realized_sum_square += _sum_square_float64(realized_delta)
        self.delta_error_sum_square += _sum_square_float64(delta_error)
        self.delta_dot += float(
            np.sum(
                np.multiply(target_delta, realized_delta, dtype=np.float64),
                dtype=np.float64,
            )
        )
        self.maximum_absolute_error = max(
            self.maximum_absolute_error,
            float(np.max(np.abs(error), initial=0.0)),
        )

        base_norm = np.linalg.norm(base, axis=1)
        realized_norm = np.linalg.norm(realized, axis=1)
        nonzero = base_norm > 1e-12
        if bool(nonzero.any()):
            relative = np.abs(realized_norm[nonzero] - base_norm[nonzero]) / base_norm[nonzero]
            self.row_norm_relative_sum_square += float(
                np.sum(relative * relative, dtype=np.float64)
            )
            self.row_norm_rows += relative.size
            self.maximum_row_norm_relative_error = max(
                self.maximum_row_norm_relative_error,
                float(np.max(relative, initial=0.0)),
            )

        original_blocks = original_raw.reshape(original_raw.shape[0], -1, type_size)
        selected_blocks = selected_raw.reshape(selected_raw.shape[0], -1, type_size)
        changed = np.any(original_blocks != selected_blocks, axis=2)
        self.total_blocks += changed.size
        self.changed_blocks += int(np.count_nonzero(changed))
        self.changed_bytes += int(np.count_nonzero(original_raw != selected_raw))
        for value, count in zip(*np.unique(choice_indices, return_counts=True), strict=True):
            label = "original" if int(value) < 0 else f"x{multipliers[int(value)]:g}"
            self.choices[label] += int(count)
        self.tracked_chunk_array_bytes_lower_bound = max(
            self.tracked_chunk_array_bytes_lower_bound, workspace_bytes
        )

    def report(self) -> dict[str, object]:
        target_approximation_rmse = math.sqrt(
            self.quantization_sse / max(self.elements, 1)
        )
        relative_l2 = (
            math.sqrt(self.quantization_sse / self.target_sum_square)
            if self.target_sum_square > 0
            else None
        )
        if self.delta_target_sum_square > 0 and self.delta_realized_sum_square > 0:
            delta_cosine = max(
                -1.0,
                min(
                    1.0,
                    self.delta_dot
                    / math.sqrt(
                        self.delta_target_sum_square
                        * self.delta_realized_sum_square
                    ),
                ),
            )
            delta_norm_ratio = math.sqrt(
                self.delta_realized_sum_square / self.delta_target_sum_square
            )
        else:
            delta_cosine = None
            delta_norm_ratio = None
        delta_relative_error = (
            math.sqrt(self.delta_error_sum_square / self.delta_target_sum_square)
            if self.delta_target_sum_square > 0
            else None
        )
        row_norm_rms = (
            math.sqrt(self.row_norm_relative_sum_square / self.row_norm_rows)
            if self.row_norm_rows
            else 0.0
        )
        return {
            "target_approximation_rmse": target_approximation_rmse,
            "target_approximation_relative_l2_error": relative_l2,
            "maximum_absolute_target_error": self.maximum_absolute_error,
            "delta_cosine": delta_cosine,
            "delta_norm_ratio": delta_norm_ratio,
            "delta_relative_error": delta_relative_error,
            "row_norm_relative_rms": row_norm_rms,
            "maximum_row_norm_relative_error": self.maximum_row_norm_relative_error,
            "total_blocks": self.total_blocks,
            "changed_blocks": self.changed_blocks,
            "changed_block_fraction": (
                self.changed_blocks / self.total_blocks if self.total_blocks else 0.0
            ),
            "changed_bytes": self.changed_bytes,
            "quantization_choices": dict(sorted(self.choices.items())),
            "tracked_chunk_array_bytes_lower_bound": (
                self.tracked_chunk_array_bytes_lower_bound
            ),
        }


@dataclass(frozen=True)
class _FileHashSnapshot:
    """Content digests tied to one stable file identity."""

    sha256: str
    untouched_sha256: str
    size_bytes: int
    device: int
    inode: int
    mtime_ns: int
    ctime_ns: int


def _resolve_plan(
    payload: bytes,
) -> tuple[
    GGUFQuantizedAblationPlan | GGUFQ8AblationPlan,
    tuple[_ResolvedEdit, ...],
    int,
    bool,
    GGUFArithmeticMode,
]:
    document = json.loads(payload)
    schema = document.get("schema_version")
    if schema in {"gguf-static-ablation-v2", "gguf-static-ablation-v3"}:
        plan = GGUFQuantizedAblationPlan.model_validate(document)
        edits = tuple(
            _ResolvedEdit(
                tensor_name=edit.tensor_name,
                expected_quantization=edit.expected_quantization,
                a_key=edit.a_key,
                b_key=edit.b_key,
                right_key=edit.right_key,
                strength=edit.strength,
                preserve_row_norms=edit.preserve_row_norms,
                preserve_original_blocks=edit.preserve_original_blocks,
                quantization_multipliers=edit.quantization_multipliers,
                minimum_block_improvement=edit.minimum_block_improvement,
                require_payload_change=edit.require_payload_change,
                minimum_delta_cosine=edit.minimum_delta_cosine,
                maximum_delta_relative_error=edit.maximum_delta_relative_error,
                maximum_row_norm_relative_error=edit.maximum_row_norm_relative_error,
            )
            for edit in plan.edits
        )
        arithmetic_mode: GGUFArithmeticMode = (
            "legacy-plan-v2"
            if schema == "gguf-static-ablation-v2"
            else "chunk-stable-v1"
        )
        return (
            plan,
            edits,
            plan.row_chunk_size,
            plan.verify_untouched_bytes,
            arithmetic_mode,
        )
    if schema == "gguf-q8-ablation-v1":
        plan = GGUFQ8AblationPlan.model_validate(document)
        edits = tuple(
            _ResolvedEdit(
                tensor_name=edit.tensor_name,
                expected_quantization="Q8_0",
                a_key=edit.a_key,
                b_key=edit.b_key,
                right_key=edit.right_key,
                strength=edit.strength,
                preserve_row_norms=edit.preserve_row_norms,
                preserve_original_blocks=False,
                quantization_multipliers=(1.0,),
                minimum_block_improvement=0.0,
                require_payload_change=edit.strength > 0,
                minimum_delta_cosine=None,
                maximum_delta_relative_error=None,
                maximum_row_norm_relative_error=None,
            )
            for edit in plan.edits
        )
        return plan, edits, 128, True, "legacy-plan-v2"
    raise ValueError(
        "unsupported GGUF edit plan schema; expected gguf-static-ablation-v3, "
        "gguf-static-ablation-v2 or gguf-q8-ablation-v1"
    )


def _matrix_factor(value: np.ndarray, *, key: str) -> np.ndarray:
    factor = np.asarray(value, dtype=np.float32)
    if factor.ndim == 1:
        factor = factor[:, None]
    if factor.ndim != 2 or min(factor.shape) < 1:
        raise ValueError(f"factor {key!r} must be a non-empty vector or matrix")
    if not np.isfinite(factor).all():
        raise ValueError(f"factor {key!r} contains a non-finite value")
    return np.ascontiguousarray(factor)


def _load_factors(
    tensor_payload: bytes,
    edits: tuple[_ResolvedEdit, ...],
) -> dict[str, np.ndarray]:
    required = {
        key
        for edit in edits
        for key in (
            edit.a_key,
            edit.right_key if edit.right_key is not None else edit.resolved_b_key,
        )
    }
    try:
        artifact = load_safetensors(tensor_payload)
    except Exception as error:
        raise RuntimeError("invalid safetensors factor artifact") from error
    missing = sorted(required - set(artifact))
    if missing:
        raise RuntimeError(f"tensor artifact is missing factors: {missing}")
    factors = {
        key: _matrix_factor(artifact[key], key=key) for key in sorted(required)
    }
    return factors


def _sha256_array(value: np.ndarray, *, chunk_rows: int = 1024) -> str:
    array = np.asarray(value).view(np.uint8).reshape(-1, value.shape[-1] * value.dtype.itemsize)
    digest = hashlib.sha256()
    for start in range(0, array.shape[0], chunk_rows):
        digest.update(memoryview(np.ascontiguousarray(array[start : start + chunk_rows])))
    return digest.hexdigest()


def _metadata_scalar(reader: Any, name: str) -> int | str | None:
    field_value = reader.fields.get(name)
    if field_value is None or not field_value.data:
        return None
    part = field_value.parts[field_value.data[0]]
    value = part[0] if getattr(part, "size", 0) else None
    if value is None:
        return None
    return value.item() if hasattr(value, "item") else value


def inspect_quantized_gguf(
    path: str | Path,
    *,
    ggml_library: str | Path | None = None,
) -> dict[str, Any]:
    """Describe same-type editable tensors in a mixed-quantization GGUF."""

    GGUFReader, _GGMLQuantizationType, _dequantize, _quantize = _gguf_api()
    source = Path(path).expanduser().resolve(strict=True)
    reader = GGUFReader(source)
    _validate_tensor_payload_layout(reader, source.stat().st_size)
    native_endian = reader.byte_order == "I"
    codec = GGUFQuantizationCodecRegistry(ggml_library=ggml_library)
    codec_errors: dict[str, str | None] = {}
    rows = []
    histogram: Counter[str] = Counter()
    for tensor in reader.tensors:
        qname = tensor.tensor_type.name
        histogram[qname] += 1
        logical_shape = tuple(int(value) for value in reversed(tensor.shape.tolist()))
        layout = QUANT_LAYOUTS.get(qname)
        reasons = []
        if not native_endian:
            reasons.append("opposite-endian GGUF")
        if layout is None:
            reasons.append("no same-type codec")
        elif qname not in codec_errors:
            try:
                codec.ensure_supported(tensor.tensor_type)
            except RuntimeError as error:
                codec_errors[qname] = str(error)
            else:
                codec_errors[qname] = None
        if layout is not None and codec_errors.get(qname) is not None:
            reasons.append(f"codec unavailable: {codec_errors[qname]}")
        if len(logical_shape) < 2:
            reasons.append("not a matrix or expert bank")
        elif layout is not None and logical_shape[-1] % layout.block_size:
            reasons.append(f"input dimension is not divisible by {layout.block_size}")
        rows.append(
            {
                "name": tensor.name,
                "shape": list(logical_shape),
                "quantization": qname,
                "quantized_bytes": int(tensor.n_bytes),
                "data_offset": int(tensor.data_offset),
                "editable": not reasons,
                "ineligible_reasons": reasons,
                "block_size": layout.block_size if layout else None,
                "type_size": layout.type_size if layout else None,
            }
        )
    file_type = _metadata_scalar(reader, "general.file_type")
    return {
        "schema_version": "gguf-quantized-inspection-v2",
        "path": str(source),
        "size_bytes": source.stat().st_size,
        "native_endian": native_endian,
        "general_file_type": file_type,
        "codec": codec.provenance(),
        "codec_availability": {
            name: {"available": error is None, "error": error}
            for name, error in sorted(codec_errors.items())
        },
        "tensor_count": len(reader.tensors),
        "editable_tensor_count": sum(bool(row["editable"]) for row in rows),
        "quantization_histogram": dict(sorted(histogram.items())),
        "tensors": rows,
    }


def _normalized_rows(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    normalized = np.zeros_like(values)
    np.divide(values, norms, out=normalized, where=norms > 0)
    return normalized, norms


def _edited_from_delta(
    base: np.ndarray,
    delta: np.ndarray,
    *,
    scale: float,
    preserve_row_norms: bool,
) -> np.ndarray:
    if preserve_row_norms:
        working, original_norms = _normalized_rows(base)
    else:
        working = base.copy()
        original_norms = None
    working -= np.float32(scale) * delta
    if original_norms is not None:
        edited_norms = np.linalg.norm(working, axis=1, keepdims=True)
        collapsed = (original_norms[:, 0] > 1e-7) & (edited_norms[:, 0] <= 1e-7)
        if bool(collapsed.any()):
            raise RuntimeError("quantized ablation collapsed a non-zero output row")
        np.divide(working, edited_norms, out=working, where=edited_norms > 0)
        working *= original_norms
    if not np.isfinite(working).all():
        raise RuntimeError("quantized ablation produced non-finite values")
    return working


def _block_error(values: np.ndarray, target: np.ndarray, block_size: int) -> np.ndarray:
    difference = np.subtract(values, target, dtype=np.float64).reshape(
        values.shape[0], -1, block_size
    )
    return np.sum(np.square(difference), axis=2, dtype=np.float64)


def _streaming_projection(
    bank: np.ndarray,
    b: np.ndarray,
    codec: GGUFQuantizationCodecRegistry,
    qtype: Any,
    input_dim: int,
    *,
    row_chunk_size: int,
    preserve_row_norms: bool,
) -> tuple[np.ndarray, int]:
    """Accumulate ``B.T @ W`` in canonical global-row reduction tiles."""

    output_dim = bank.shape[0]
    rank = b.shape[1]
    reduction_rows = min(_PROJECTION_REDUCTION_ROWS, output_dim)
    pending_b = np.empty((reduction_rows, rank), dtype=np.float32)
    pending_values = np.empty((reduction_rows, input_dim), dtype=np.float32)
    pending_count = 0
    projection = np.zeros((rank, input_dim), dtype=np.float32)
    workspace_bytes = projection.nbytes + pending_b.nbytes + pending_values.nbytes

    for start in range(0, output_dim, row_chunk_size):
        stop = min(start + row_chunk_size, output_dim)
        base = codec.dequantize_rows(bank[start:stop], qtype, input_dim)
        projected = _normalized_rows(base)[0] if preserve_row_norms else base
        workspace_bytes = max(
            workspace_bytes,
            projection.nbytes
            + pending_b.nbytes
            + pending_values.nbytes
            + base.nbytes
            + (0 if projected is base else projected.nbytes),
        )
        offset = 0
        while offset < projected.shape[0]:
            copied = min(
                reduction_rows - pending_count,
                projected.shape[0] - offset,
            )
            pending_b[pending_count : pending_count + copied] = b[
                start + offset : start + offset + copied
            ]
            pending_values[pending_count : pending_count + copied] = projected[
                offset : offset + copied
            ]
            pending_count += copied
            offset += copied
            if pending_count == reduction_rows:
                projection += pending_b.T @ pending_values
                pending_count = 0
    if pending_count:
        projection += (
            pending_b[:pending_count].T @ pending_values[:pending_count]
        )
    return projection, workspace_bytes


def _legacy_streaming_projection(
    bank: np.ndarray,
    b: np.ndarray,
    codec: GGUFQuantizationCodecRegistry,
    qtype: Any,
    input_dim: int,
    *,
    row_chunk_size: int,
    preserve_row_norms: bool,
) -> tuple[np.ndarray, int]:
    """Replay the chunk-dependent arithmetic shipped with v2 plans."""

    output_dim = bank.shape[0]
    projection = np.zeros((b.shape[1], input_dim), dtype=np.float32)
    workspace_bytes = projection.nbytes
    for start in range(0, output_dim, row_chunk_size):
        stop = min(start + row_chunk_size, output_dim)
        base = codec.dequantize_rows(bank[start:stop], qtype, input_dim)
        projected = _normalized_rows(base)[0] if preserve_row_norms else base
        projection += b[start:stop].T @ projected
        workspace_bytes = max(
            workspace_bytes,
            projection.nbytes
            + base.nbytes
            + (0 if projected is base else projected.nbytes),
        )
    return projection, workspace_bytes


def _edit_tensor_payload(
    tensor: Any,
    edit: _ResolvedEdit,
    factors: dict[str, np.ndarray],
    codec: GGUFQuantizationCodecRegistry,
    *,
    row_chunk_size: int,
    arithmetic_mode: GGUFArithmeticMode = "chunk-stable-v1",
) -> dict[str, object]:
    if arithmetic_mode not in {"legacy-plan-v2", "chunk-stable-v1"}:
        raise ValueError(f"unsupported GGUF arithmetic mode: {arithmetic_mode}")
    qtype = tensor.tensor_type
    layout = QUANT_LAYOUTS[qtype.name]
    logical_shape = tuple(int(value) for value in reversed(tensor.shape.tolist()))
    matrix_count = int(np.prod(logical_shape[:-2])) or 1
    output_dim, input_dim = logical_shape[-2:]
    encoded_row_bytes = input_dim // layout.block_size * layout.type_size
    raw = np.asarray(tensor.data)
    if raw.dtype.itemsize != 1 or raw.shape[-1] != encoded_row_bytes:
        raise RuntimeError(
            f"unexpected encoded layout for {tensor.name}: {raw.shape}/{raw.dtype}; "
            f"expected final byte dimension {encoded_row_bytes}"
        )
    encoded_banks = raw.view(np.uint8).reshape(
        matrix_count, output_dim, encoded_row_bytes
    )
    before_sha256 = _sha256_array(raw)
    total_expected_blocks = matrix_count * output_dim * (input_dim // layout.block_size)
    if edit.strength == 0:
        return {
            "before_payload_sha256": before_sha256,
            "after_payload_sha256": before_sha256,
            "payload_changed": False,
            "quantization_metrics": {
                "target_approximation_rmse": 0.0,
                "target_approximation_relative_l2_error": 0.0,
                "maximum_absolute_target_error": 0.0,
                "delta_cosine": None,
                "delta_norm_ratio": None,
                "delta_relative_error": None,
                "row_norm_relative_rms": 0.0,
                "maximum_row_norm_relative_error": 0.0,
                "total_blocks": total_expected_blocks,
                "changed_blocks": 0,
                "changed_block_fraction": 0.0,
                "changed_bytes": 0,
                "quantization_choices": {"original": total_expected_blocks},
                "tracked_chunk_array_bytes_lower_bound": 0,
            },
        }

    a = factors[edit.a_key]
    right = factors[edit.right_key] if edit.right_key is not None else None
    b = factors[edit.resolved_b_key] if right is None else None
    accumulator = _ErrorAccumulator()

    for bank_index in range(matrix_count):
        bank = encoded_banks[bank_index]
        projection: np.ndarray | None = None
        projection_workspace_bytes = 0
        if right is None:
            assert b is not None
            projection_function = (
                _legacy_streaming_projection
                if arithmetic_mode == "legacy-plan-v2"
                else _streaming_projection
            )
            projection, projection_workspace_bytes = projection_function(
                bank,
                b,
                codec,
                qtype,
                input_dim,
                row_chunk_size=row_chunk_size,
                preserve_row_norms=edit.preserve_row_norms,
            )

        for start in range(0, output_dim, row_chunk_size):
            stop = min(start + row_chunk_size, output_dim)
            original_raw = np.array(bank[start:stop], copy=True)
            base = codec.dequantize_rows(original_raw, qtype, input_dim)
            if right is None:
                assert projection is not None
                delta = (
                    a[start:stop] @ projection
                    if arithmetic_mode == "legacy-plan-v2"
                    else np.matmul(a[start:stop], projection, dtype=np.float32)
                )
            else:
                delta = (
                    a[start:stop] @ right.T
                    if arithmetic_mode == "legacy-plan-v2"
                    else np.matmul(a[start:stop], right.T, dtype=np.float32)
                )
            target = _edited_from_delta(
                base,
                delta,
                scale=edit.strength,
                preserve_row_norms=edit.preserve_row_norms,
            )
            blocks_per_row = input_dim // layout.block_size
            if edit.preserve_original_blocks:
                selected_raw = original_raw.copy()
                best_error = _block_error(base, target, layout.block_size)
                choices = np.full((base.shape[0], blocks_per_row), -1, dtype=np.int16)
            else:
                selected_raw = np.empty_like(original_raw)
                best_error = np.full(
                    (base.shape[0], blocks_per_row), np.inf, dtype=np.float64
                )
                choices = np.full((base.shape[0], blocks_per_row), -2, dtype=np.int16)

            workspace_bytes = max(
                projection_workspace_bytes,
                base.nbytes
                + delta.nbytes
                + target.nbytes
                + original_raw.nbytes
                + selected_raw.nbytes
                + best_error.nbytes
                + choices.nbytes
                + (projection.nbytes if projection is not None else 0),
            )
            for multiplier_index, multiplier in enumerate(edit.quantization_multipliers):
                candidate = (
                    target
                    if multiplier == 1.0
                    else _edited_from_delta(
                        base,
                        delta,
                        scale=edit.strength * multiplier,
                        preserve_row_norms=edit.preserve_row_norms,
                    )
                )
                encoded_candidate = codec.quantize_rows(candidate, qtype)
                if encoded_candidate.shape != original_raw.shape:
                    raise RuntimeError(
                        f"same-type codec changed {tensor.name} row layout: "
                        f"{encoded_candidate.shape} != {original_raw.shape}"
                    )
                realized_candidate = codec.dequantize_rows(
                    encoded_candidate, qtype, input_dim
                )
                candidate_error = _block_error(
                    realized_candidate, target, layout.block_size
                )
                threshold = best_error * (1.0 - edit.minimum_block_improvement)
                better = candidate_error < threshold
                if bool(better.any()):
                    selected_blocks = selected_raw.reshape(
                        selected_raw.shape[0], blocks_per_row, layout.type_size
                    )
                    candidate_blocks = encoded_candidate.reshape(
                        encoded_candidate.shape[0], blocks_per_row, layout.type_size
                    )
                    selected_blocks[better] = candidate_blocks[better]
                    best_error[better] = candidate_error[better]
                    choices[better] = multiplier_index
                workspace_bytes = max(
                    workspace_bytes,
                    base.nbytes
                    + delta.nbytes
                    + target.nbytes
                    + candidate.nbytes
                    + encoded_candidate.nbytes
                    + realized_candidate.nbytes
                    + selected_raw.nbytes
                    + original_raw.nbytes
                    + best_error.nbytes
                    + choices.nbytes
                    + (projection.nbytes if projection is not None else 0),
                    projection_workspace_bytes,
                )

            if bool((choices == -2).any()):
                raise RuntimeError("no quantization candidate was selected for one or more blocks")
            realized = codec.dequantize_rows(selected_raw, qtype, input_dim)
            accumulator.update(
                base=base,
                target=target,
                realized=realized,
                original_raw=original_raw,
                selected_raw=selected_raw,
                choice_indices=choices,
                multipliers=edit.quantization_multipliers,
                type_size=layout.type_size,
                workspace_bytes=max(workspace_bytes, realized.nbytes),
            )
            bank[start:stop] = selected_raw

    after_sha256 = _sha256_array(raw)
    metrics = accumulator.report()
    changed = before_sha256 != after_sha256
    if edit.require_payload_change and not changed:
        raise RuntimeError(
            f"edit for {tensor.name} produced no quantized payload change; "
            "increase the strength or explicitly disable require_payload_change"
        )
    delta_cosine = metrics["delta_cosine"]
    if edit.minimum_delta_cosine is not None and (
        delta_cosine is None or float(delta_cosine) < edit.minimum_delta_cosine
    ):
        raise RuntimeError(
            f"edit for {tensor.name} failed its delta cosine gate: "
            f"{delta_cosine} < {edit.minimum_delta_cosine}"
        )
    delta_relative_error = metrics["delta_relative_error"]
    if edit.maximum_delta_relative_error is not None and (
        delta_relative_error is None
        or float(delta_relative_error) > edit.maximum_delta_relative_error
    ):
        raise RuntimeError(
            f"edit for {tensor.name} failed its delta error gate: "
            f"{delta_relative_error} > {edit.maximum_delta_relative_error}"
        )
    norm_error = float(metrics["maximum_row_norm_relative_error"])
    if (
        edit.maximum_row_norm_relative_error is not None
        and norm_error > edit.maximum_row_norm_relative_error
    ):
        raise RuntimeError(
            f"edit for {tensor.name} failed its row-norm gate: "
            f"{norm_error} > {edit.maximum_row_norm_relative_error}"
        )
    return {
        "before_payload_sha256": before_sha256,
        "after_payload_sha256": after_sha256,
        "payload_changed": changed,
        "quantization_metrics": metrics,
    }


def _validated_intervals(
    file_size: int,
    intervals: list[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    ordered = tuple(sorted(intervals))
    previous_stop = 0
    for start, stop in ordered:
        if start < 0 or stop <= start or stop > file_size:
            raise RuntimeError(f"invalid GGUF tensor payload interval: {(start, stop)}")
        if start < previous_stop:
            raise RuntimeError("GGUF target tensor payload intervals overlap")
        previous_stop = stop
    return ordered


def _validate_tensor_payload_layout(reader: Any, file_size: int) -> None:
    intervals = sorted(
        (
            int(tensor.data_offset),
            int(tensor.data_offset + tensor.n_bytes),
            tensor.name,
        )
        for tensor in reader.tensors
    )
    previous_stop = 0
    previous_name = "<header>"
    for start, stop, name in intervals:
        if start < 0 or stop <= start or stop > file_size:
            raise RuntimeError(
                f"invalid GGUF payload interval for {name}: {(start, stop)}"
            )
        if start < previous_stop:
            raise RuntimeError(
                f"overlapping GGUF tensor payloads: {previous_name} and {name}"
            )
        previous_stop = stop
        previous_name = name


def _file_and_untouched_sha256(
    path: Path,
    intervals: tuple[tuple[int, int], ...],
    *,
    chunk_size: int = 8 * 1024 * 1024,
) -> _FileHashSnapshot:
    """Hash the complete file and all untouched regions in one sequential pass."""

    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    full_digest = hashlib.sha256()
    untouched_digest = hashlib.sha256()
    with path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        file_size = before.st_size
        ordered_intervals = _validated_intervals(file_size, list(intervals))
        cursor = 0
        for start, stop in (*ordered_intervals, (file_size, file_size)):
            untouched_remaining = start - cursor
            while untouched_remaining:
                chunk = handle.read(min(chunk_size, untouched_remaining))
                if not chunk:
                    raise RuntimeError("GGUF ended while hashing untouched bytes")
                full_digest.update(chunk)
                untouched_digest.update(chunk)
                untouched_remaining -= len(chunk)
            targeted_remaining = stop - start
            while targeted_remaining:
                chunk = handle.read(min(chunk_size, targeted_remaining))
                if not chunk:
                    raise RuntimeError("GGUF ended while hashing a target payload")
                full_digest.update(chunk)
                targeted_remaining -= len(chunk)
            cursor = stop
        if handle.read(1):
            raise RuntimeError("GGUF grew while it was being hashed")
        after = os.fstat(handle.fileno())
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if identity_after != identity_before:
        raise RuntimeError("GGUF changed while it was being hashed")
    return _FileHashSnapshot(
        sha256=full_digest.hexdigest(),
        untouched_sha256=untouched_digest.hexdigest(),
        size_bytes=after.st_size,
        device=after.st_dev,
        inode=after.st_ino,
        mtime_ns=after.st_mtime_ns,
        ctime_ns=after.st_ctime_ns,
    )


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        # Some platforms/filesystems do not support fsync on directory handles.
        pass
    finally:
        os.close(descriptor)


def _publish_output(
    temporary: Path,
    output: Path,
    *,
    force: bool,
    expected: _FileHashSnapshot | None = None,
) -> str:
    """Publish atomically and prove the already-hashed inode was installed."""

    if expected is None:
        expected = _file_and_untouched_sha256(temporary, ())
    temporary_stat = temporary.stat()
    if (
        temporary_stat.st_dev != expected.device
        or temporary_stat.st_ino != expected.inode
        or temporary_stat.st_size != expected.size_bytes
        or temporary_stat.st_mtime_ns != expected.mtime_ns
    ):
        raise RuntimeError("temporary GGUF changed after final hashing")
    if force:
        os.replace(temporary, output)
    else:
        try:
            os.link(temporary, output)
        except FileExistsError as error:
            raise FileExistsError(
                f"refusing to overwrite concurrently-created output: {output}"
            ) from error
        temporary.unlink()
    _fsync_directory(output.parent)
    output_stat = output.stat()
    if (
        output_stat.st_dev != expected.device
        or output_stat.st_ino != expected.inode
        or output_stat.st_size != expected.size_bytes
        or output_stat.st_mtime_ns != expected.mtime_ns
    ):
        raise RuntimeError("published GGUF is not the verified temporary inode")
    return expected.sha256


def apply_quantized_gguf_ablation(
    source_path: str | Path,
    output_path: str | Path | None,
    plan_path: str | Path,
    tensor_path: str | Path,
    *,
    dry_run: bool = False,
    force: bool = False,
    ggml_library: str | Path | None = None,
) -> dict[str, Any]:
    """Apply a type-preserving plan to Q2_K..Q6_K and common GGUF quants."""

    started = time.perf_counter()
    GGUFReader, _GGMLQuantizationType, _dequantize, _quantize = _gguf_api()
    source = Path(source_path).expanduser().resolve(strict=True)
    plan_file = Path(plan_path).expanduser().resolve(strict=True)
    tensor_file = Path(tensor_path).expanduser().resolve(strict=True)
    output = (
        Path(output_path).expanduser().resolve(strict=False)
        if output_path is not None
        else None
    )
    if not source.is_file() or not plan_file.is_file() or not tensor_file.is_file():
        raise ValueError("source, plan, and tensor artifact must all be files")
    if output is not None and output == source:
        raise ValueError("source and output GGUF paths must differ")
    if output is not None and output.is_dir():
        raise IsADirectoryError(f"output GGUF path is a directory: {output}")
    if not dry_run and output is None:
        raise ValueError("an output GGUF path is required unless --dry-run is used")
    if output is not None and output.exists() and not force:
        raise FileExistsError(f"refusing to overwrite existing output: {output}")

    plan_payload = plan_file.read_bytes()
    plan_sha256 = hashlib.sha256(plan_payload).hexdigest()
    (
        plan,
        edits,
        row_chunk_size,
        verify_untouched,
        arithmetic_mode,
    ) = _resolve_plan(plan_payload)
    # A real merge validates the immutable copied snapshot below. Hashing the
    # mutable source first would add a complete, redundant multi-gigabyte scan.
    # Dry-runs have no snapshot, so they still validate the source directly.
    source_sha256 = plan.source_sha256
    if dry_run:
        source_sha256 = sha256_file(source)
        if source_sha256 != plan.source_sha256:
            raise RuntimeError(
                f"source GGUF hash mismatch: {source_sha256} != {plan.source_sha256}"
            )
    tensor_payload = tensor_file.read_bytes()
    tensor_sha256 = hashlib.sha256(tensor_payload).hexdigest()
    if tensor_sha256 != plan.tensor_artifact_sha256:
        raise RuntimeError(
            f"factor artifact hash mismatch: {tensor_sha256} != {plan.tensor_artifact_sha256}"
        )
    factors = _load_factors(tensor_payload, edits)

    reader = GGUFReader(source)
    _validate_tensor_payload_layout(reader, source.stat().st_size)
    if reader.byte_order != "I":
        raise RuntimeError("direct quantized editing refuses opposite-endian GGUF files")
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    missing = sorted(edit.tensor_name for edit in edits if edit.tensor_name not in tensors)
    if missing:
        raise RuntimeError(f"GGUF is missing target tensors: {missing}")

    codec = GGUFQuantizationCodecRegistry(ggml_library=ggml_library)
    prepared = []
    for edit in edits:
        tensor = tensors[edit.tensor_name]
        qname = tensor.tensor_type.name
        if qname != edit.expected_quantization:
            raise RuntimeError(
                f"target {edit.tensor_name} is {qname}, not "
                f"{edit.expected_quantization} (the plan's expected quantization)"
            )
        codec.ensure_supported(tensor.tensor_type)
        logical_shape = tuple(int(value) for value in reversed(tensor.shape.tolist()))
        if len(logical_shape) < 2:
            raise RuntimeError(f"target {edit.tensor_name} is not a matrix or expert bank")
        layout = QUANT_LAYOUTS[qname]
        if logical_shape[-1] % layout.block_size:
            raise RuntimeError(
                f"target {edit.tensor_name} input dimension {logical_shape[-1]} is not "
                f"divisible by the {qname} block size {layout.block_size}"
            )
        a = factors[edit.a_key]
        output_dim = logical_shape[-2]
        if a.shape[0] != output_dim:
            raise RuntimeError(
                f"factor dimension mismatch for {edit.tensor_name}: "
                f"expected output {output_dim}, got A={a.shape[0]}"
            )
        if edit.right_key is None:
            b = factors[edit.resolved_b_key]
            if b.shape != a.shape:
                raise RuntimeError(
                    f"projector factors differ for {edit.tensor_name}: A={a.shape}, B={b.shape}"
                )
        else:
            right = factors[edit.right_key]
            if right.shape != (logical_shape[-1], a.shape[1]):
                raise RuntimeError(
                    f"direct right factor mismatch for {edit.tensor_name}: "
                    f"A={a.shape}, right={right.shape}, input={logical_shape[-1]}"
                )
        prepared.append(
            {
                "tensor_name": edit.tensor_name,
                "logical_shape": list(logical_shape),
                "quantization": qname,
                "codec_backend": codec.backend_for(tensor.tensor_type),
                "matrix_count": int(np.prod(logical_shape[:-2])) or 1,
                "rank": int(a.shape[1]),
                "edit_mode": "direct-low-rank" if edit.right_key else "projector",
                "strength": edit.strength,
                "preserve_row_norms": edit.preserve_row_norms,
                "preserve_original_blocks": edit.preserve_original_blocks,
                "quantization_multipliers": list(edit.quantization_multipliers),
                "data_offset": int(tensor.data_offset),
                "quantized_bytes": int(tensor.n_bytes),
            }
        )

    report: dict[str, Any] = {
        "schema_version": "gguf-quantized-static-merge-report-v3",
        "source": {
            "path": str(source),
            "sha256": source_sha256,
            "size_bytes": source.stat().st_size,
        },
        "plan": {
            "path": str(plan_file),
            "sha256": plan_sha256,
            "schema_version": plan.schema_version,
        },
        "tensor_artifact": {"path": str(tensor_file), "sha256": tensor_sha256},
        "codec": codec.provenance(),
        "row_chunk_size": row_chunk_size,
        "arithmetic_mode": arithmetic_mode,
        "verify_untouched_bytes": verify_untouched,
        "search_only": not verify_untouched,
        "dry_run": dry_run,
        "edits": prepared,
    }
    if dry_run:
        report["elapsed_seconds"] = time.perf_counter() - started
        return report

    del tensors
    del reader
    gc.collect()
    assert output is not None
    output.parent.mkdir(parents=True, exist_ok=True)
    required_bytes = source.stat().st_size + 64 * 1024 * 1024
    free_bytes = shutil.disk_usage(output.parent).free
    if free_bytes < required_bytes:
        raise RuntimeError(
            f"insufficient free space: need at least {required_bytes} bytes, have {free_bytes}"
        )

    temporary_fd, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    os.close(temporary_fd)
    temporary = Path(temporary_name)
    try:
        _copy_source(source, temporary)
        intervals = _validated_intervals(
            temporary.stat().st_size,
            [
                (
                    int(row["data_offset"]),
                    int(row["data_offset"]) + int(row["quantized_bytes"]),
                )
                for row in prepared
            ],
        )
        snapshot_hashes = _file_and_untouched_sha256(
            temporary,
            intervals if verify_untouched else (),
        )
        if snapshot_hashes.sha256 != source_sha256:
            raise RuntimeError(
                "source GGUF changed while its immutable edit snapshot was created: "
                f"{snapshot_hashes.sha256} != {source_sha256}"
            )
        untouched_before = (
            snapshot_hashes.untouched_sha256
            if verify_untouched
            else None
        )
        writable = GGUFReader(temporary, mode="r+")
        _validate_tensor_payload_layout(writable, temporary.stat().st_size)
        if writable.byte_order != "I":
            raise RuntimeError("temporary GGUF changed byte order unexpectedly")
        writable_tensors = {tensor.name: tensor for tensor in writable.tensors}
        result_rows = []
        for edit, prepared_row in zip(edits, prepared, strict=True):
            tensor = writable_tensors[edit.tensor_name]
            edit_started = time.perf_counter()
            payload = _edit_tensor_payload(
                tensor,
                edit,
                factors,
                codec,
                row_chunk_size=row_chunk_size,
                arithmetic_mode=arithmetic_mode,
            )
            result_rows.append(
                {
                    **prepared_row,
                    **payload,
                    "elapsed_seconds": time.perf_counter() - edit_started,
                }
            )
        writable.data.flush()
        del tensor
        del writable_tensors
        del writable
        gc.collect()

        descriptor = os.open(temporary, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

        reopened = GGUFReader(temporary)
        _validate_tensor_payload_layout(reopened, temporary.stat().st_size)
        reopened_tensors = {tensor.name: tensor for tensor in reopened.tensors}
        for prepared_row, result_row in zip(prepared, result_rows, strict=True):
            tensor = reopened_tensors[prepared_row["tensor_name"]]
            if (
                tensor.tensor_type.name != prepared_row["quantization"]
                or int(tensor.data_offset) != prepared_row["data_offset"]
                or int(tensor.n_bytes) != prepared_row["quantized_bytes"]
                or _sha256_array(tensor.data) != result_row["after_payload_sha256"]
            ):
                raise RuntimeError(
                    f"post-write GGUF validation failed for {prepared_row['tensor_name']}"
                )
        del tensor
        del reopened_tensors
        del reopened
        gc.collect()

        final_hashes = _file_and_untouched_sha256(
            temporary,
            intervals if verify_untouched else (),
        )
        if verify_untouched:
            if final_hashes.untouched_sha256 != untouched_before:
                raise RuntimeError("GGUF bytes changed outside declared tensor payloads")
            report["untouched_bytes_sha256"] = final_hashes.untouched_sha256
        report["untouched_bytes_verified"] = verify_untouched
        published_sha256 = _publish_output(
            temporary,
            output,
            force=force,
            expected=final_hashes,
        )
        report["dry_run"] = False
        report["edits"] = result_rows
        report["output"] = {
            "path": str(output),
            "sha256": published_sha256,
            "size_bytes": output.stat().st_size,
        }
        report["elapsed_seconds"] = time.perf_counter() - started
        return report
    finally:
        if temporary.exists():
            temporary.unlink()
