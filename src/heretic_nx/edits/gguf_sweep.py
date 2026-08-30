"""Exact multi-strength sweeps for type-preserving GGUF edits.

The ordinary GGUF editor intentionally treats each output as an independent
transaction.  That is the right default for a frozen release, but it repeats
the expensive source dequantization and projector reduction for every beta in
a search portfolio.  This module shares only arithmetic that is invariant
across plans and keeps quantization, diagnostics, integrity checks, and output
publication independent for every candidate.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import gc
import hashlib
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .gguf_codecs import GGUFQuantizationCodecRegistry, QUANT_LAYOUTS
from .gguf_q8 import _copy_source, _gguf_api
from .gguf_quant import (
    GGUFDiagnosticsMode,
    GGUFQuantizedAblationPlan,
    _ErrorAccumulator,
    _FileHashSnapshot,
    _ResolvedEdit,
    _SearchAccumulator,
    _block_error,
    _edited_from_prepared,
    _file_and_untouched_sha256,
    _load_factors,
    _prepared_edit_base,
    _publish_output,
    _resolve_plan,
    _streaming_projection,
    _validate_tensor_payload_layout,
    _validated_intervals,
)


class GGUFStrengthSweepCandidate(BaseModel):
    """One independently attested output in a shared-strength sweep."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    label: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._-]+$")
    plan_path: Path
    output_path: Path


@dataclass
class _CandidateContext:
    spec: GGUFStrengthSweepCandidate
    plan_path: Path
    output_path: Path
    plan_sha256: str
    plan: GGUFQuantizedAblationPlan
    edits: tuple[_ResolvedEdit, ...]
    temporary: Path | None = None
    reader: Any | None = None
    result_rows: list[dict[str, object]] | None = None
    final_hashes: _FileHashSnapshot | None = None
    snapshot_copy_mode: str | None = None


@dataclass
class _BoundSource:
    """An inode-stable path held independently of the caller's pathname."""

    path: Path
    mode: str
    descriptor: int
    hardlink: Path | None = None

    def close(self) -> None:
        if self.hardlink is not None:
            try:
                self.hardlink.unlink()
            except FileNotFoundError:
                pass
            self.hardlink = None
        if self.descriptor >= 0:
            os.close(self.descriptor)
            self.descriptor = -1


def _bind_source_snapshot(source: Path, parents: Sequence[Path]) -> _BoundSource:
    """Bind a source inode before any reader, hash, or snapshot copy opens it.

    A verified hard link preserves clone support. If no candidate filesystem
    can host one, ``/dev/fd`` still gives every consumer the exact open inode;
    a pathname replacement can no longer splice two GGUF versions together.
    """

    descriptor = os.open(source, os.O_RDONLY)
    opened = os.fstat(descriptor)
    unique_parents = list(dict.fromkeys((*parents, source.parent)))
    for parent in unique_parents:
        hardlink: Path | None = None
        try:
            parent.mkdir(parents=True, exist_ok=True)
            temporary_fd, temporary_name = tempfile.mkstemp(
                prefix=f".{source.name}.heretic-source-",
                suffix=".link",
                dir=parent,
            )
            os.close(temporary_fd)
            hardlink = Path(temporary_name)
            hardlink.unlink()
            os.link(source, hardlink)
            linked = hardlink.stat()
            if linked.st_dev == opened.st_dev and linked.st_ino == opened.st_ino:
                return _BoundSource(
                    path=hardlink,
                    mode="hardlink",
                    descriptor=descriptor,
                    hardlink=hardlink,
                )
            hardlink.unlink()
        except OSError:
            try:
                if hardlink is not None:
                    hardlink.unlink()
            except FileNotFoundError:
                pass

    descriptor_path = Path(f"/dev/fd/{descriptor}")
    try:
        linked = descriptor_path.stat()
    except OSError as error:
        os.close(descriptor)
        raise RuntimeError(
            "unable to create an inode-stable GGUF source binding"
        ) from error
    if linked.st_dev != opened.st_dev or linked.st_ino != opened.st_ino:
        os.close(descriptor)
        raise RuntimeError("open GGUF descriptor changed identity unexpectedly")
    return _BoundSource(
        path=descriptor_path,
        mode="open-file-descriptor",
        descriptor=descriptor,
    )


@dataclass
class _PayloadState:
    edit: _ResolvedEdit
    accumulator: _ErrorAccumulator | _SearchAccumulator | None
    after_digest: Any


def _stable_source_identity(snapshot: _FileHashSnapshot, path: Path) -> bool:
    current = path.stat()
    return (
        current.st_dev == snapshot.device
        and current.st_ino == snapshot.inode
        and current.st_size == snapshot.size_bytes
        and current.st_mtime_ns == snapshot.mtime_ns
        and current.st_ctime_ns == snapshot.ctime_ns
    )


def _edit_invariant(edit: _ResolvedEdit) -> tuple[object, ...]:
    """Return arithmetic fields invariant across a strength sweep."""

    return (
        edit.tensor_name,
        edit.expected_quantization,
        edit.a_key,
        edit.b_key,
        edit.right_key,
        edit.preserve_row_norms,
        edit.preserve_original_blocks,
        edit.quantization_multipliers,
        edit.minimum_block_improvement,
        edit.minimum_delta_cosine,
        edit.maximum_delta_relative_error,
        edit.maximum_row_norm_relative_error,
    )


def _zero_strength_payload(
    raw: np.ndarray,
    *,
    total_blocks: int,
    diagnostics_mode: GGUFDiagnosticsMode,
) -> dict[str, object]:
    digest = hashlib.sha256(memoryview(np.ascontiguousarray(raw))).hexdigest()
    return {
        "before_payload_sha256": digest,
        "after_payload_sha256": digest,
        "payload_changed": False,
        "diagnostics_complete": diagnostics_mode == "full",
        "quantization_metrics": {
            "target_approximation_rmse": 0.0,
            "target_approximation_relative_l2_error": 0.0,
            "maximum_absolute_target_error": 0.0,
            "delta_cosine": None,
            "delta_norm_ratio": None,
            "delta_relative_error": None,
            "row_norm_relative_rms": 0.0,
            "maximum_row_norm_relative_error": 0.0,
            "total_blocks": total_blocks,
            "changed_blocks": 0,
            "changed_block_fraction": 0.0,
            "changed_bytes": 0,
            "quantization_choices": {"original": total_blocks},
            "tracked_chunk_array_bytes_lower_bound": 0,
        },
    }


def _finish_payload(
    tensor_name: str,
    state: _PayloadState,
    *,
    before_sha256: str,
    diagnostics_mode: GGUFDiagnosticsMode,
) -> dict[str, object]:
    assert state.accumulator is not None
    metrics = state.accumulator.report()
    after_sha256 = state.after_digest.hexdigest()
    changed = before_sha256 != after_sha256
    edit = state.edit
    if edit.require_payload_change and not changed:
        raise RuntimeError(
            f"edit for {tensor_name} produced no quantized payload change; "
            "increase the strength or explicitly disable require_payload_change"
        )
    delta_cosine = metrics["delta_cosine"]
    if edit.minimum_delta_cosine is not None and (
        delta_cosine is None or float(delta_cosine) < edit.minimum_delta_cosine
    ):
        raise RuntimeError(
            f"edit for {tensor_name} failed its delta cosine gate: "
            f"{delta_cosine} < {edit.minimum_delta_cosine}"
        )
    delta_relative_error = metrics["delta_relative_error"]
    if edit.maximum_delta_relative_error is not None and (
        delta_relative_error is None
        or float(delta_relative_error) > edit.maximum_delta_relative_error
    ):
        raise RuntimeError(
            f"edit for {tensor_name} failed its delta error gate: "
            f"{delta_relative_error} > {edit.maximum_delta_relative_error}"
        )
    if edit.maximum_row_norm_relative_error is not None:
        norm_error = float(metrics["maximum_row_norm_relative_error"])
        if norm_error > edit.maximum_row_norm_relative_error:
            raise RuntimeError(
                f"edit for {tensor_name} failed its row-norm gate: "
                f"{norm_error} > {edit.maximum_row_norm_relative_error}"
            )
    return {
        "before_payload_sha256": before_sha256,
        "after_payload_sha256": after_sha256,
        "payload_changed": changed,
        "diagnostics_complete": diagnostics_mode == "full",
        "quantization_metrics": metrics,
    }


def _edit_tensor_strength_sweep(
    source_tensor: Any,
    destination_tensors: Sequence[Any],
    edits: Sequence[_ResolvedEdit],
    factors: dict[str, np.ndarray],
    codec: GGUFQuantizationCodecRegistry,
    *,
    row_chunk_size: int,
    diagnostics_mode: GGUFDiagnosticsMode,
) -> list[dict[str, object]]:
    """Edit one tensor for many strengths while decoding its source once."""

    if len(destination_tensors) != len(edits) or not edits:
        raise ValueError("destination tensors and sweep edits must be non-empty and aligned")
    if diagnostics_mode not in {"full", "search"}:
        raise ValueError(f"unsupported GGUF diagnostics mode: {diagnostics_mode}")
    if any(_edit_invariant(edit) != _edit_invariant(edits[0]) for edit in edits[1:]):
            raise ValueError("strength sweep edits use incompatible arithmetic")
    if diagnostics_mode == "search" and any(
        gate is not None
        for edit in edits
        for gate in (
            edit.minimum_delta_cosine,
            edit.maximum_delta_relative_error,
            edit.maximum_row_norm_relative_error,
        )
    ):
        raise ValueError("search diagnostics cannot evaluate edit metric gates")

    template = edits[0]
    qtype = source_tensor.tensor_type
    layout = QUANT_LAYOUTS[qtype.name]
    logical_shape = tuple(int(value) for value in reversed(source_tensor.shape.tolist()))
    matrix_count = int(np.prod(logical_shape[:-2])) or 1
    output_dim, input_dim = logical_shape[-2:]
    encoded_row_bytes = input_dim // layout.block_size * layout.type_size
    raw = np.asarray(source_tensor.data)
    if raw.dtype.itemsize != 1 or raw.shape[-1] != encoded_row_bytes:
        raise RuntimeError(
            f"unexpected encoded layout for {source_tensor.name}: {raw.shape}/{raw.dtype}; "
            f"expected final byte dimension {encoded_row_bytes}"
        )
    source_banks = raw.view(np.uint8).reshape(
        matrix_count, output_dim, encoded_row_bytes
    )
    destination_banks: list[np.ndarray] = []
    for destination in destination_tensors:
        destination_raw = np.asarray(destination.data)
        if (
            destination.tensor_type.name != qtype.name
            or destination_raw.shape != raw.shape
            or destination_raw.dtype != raw.dtype
        ):
            raise RuntimeError(
                f"sweep destination layout changed for {source_tensor.name}"
            )
        destination_banks.append(
            destination_raw.view(np.uint8).reshape(
                matrix_count, output_dim, encoded_row_bytes
            )
        )

    blocks_per_row = input_dim // layout.block_size
    total_blocks = matrix_count * output_dim * blocks_per_row
    if not any(edit.strength > 0 for edit in edits):
        payload = _zero_strength_payload(
            raw,
            total_blocks=total_blocks,
            diagnostics_mode=diagnostics_mode,
        )
        return [dict(payload) for _ in edits]

    a = factors[template.a_key]
    right = factors[template.right_key] if template.right_key is not None else None
    b = factors[template.resolved_b_key] if right is None else None
    states = [
        _PayloadState(
            edit=edit,
            accumulator=(
                None
                if edit.strength == 0
                else (_ErrorAccumulator() if diagnostics_mode == "full" else _SearchAccumulator())
            ),
            after_digest=hashlib.sha256(),
        )
        for edit in edits
    ]
    before_digest = hashlib.sha256()

    for bank_index in range(matrix_count):
        source_bank = source_banks[bank_index]
        projection: np.ndarray | None = None
        projection_workspace_bytes = 0
        if right is None:
            assert b is not None
            projection, projection_workspace_bytes = _streaming_projection(
                source_bank,
                b,
                codec,
                qtype,
                input_dim,
                row_chunk_size=row_chunk_size,
                preserve_row_norms=template.preserve_row_norms,
            )

        for start in range(0, output_dim, row_chunk_size):
            stop = min(start + row_chunk_size, output_dim)
            original_raw = np.array(source_bank[start:stop], copy=True)
            before_digest.update(memoryview(original_raw))
            base = codec.dequantize_rows(original_raw, qtype, input_dim)
            if right is None:
                assert projection is not None
                delta = np.matmul(a[start:stop], projection, dtype=np.float32)
            else:
                delta = np.matmul(a[start:stop], right.T, dtype=np.float32)
            prepared_base, original_norms = _prepared_edit_base(
                base,
                preserve_row_norms=template.preserve_row_norms,
            )

            for state, candidate_banks in zip(
                states, destination_banks, strict=True
            ):
                destination_bank = candidate_banks[bank_index]
                edit = state.edit
                if edit.strength == 0:
                    destination_bank[start:stop] = original_raw
                    state.after_digest.update(memoryview(original_raw))
                    continue

                target = _edited_from_prepared(
                    prepared_base,
                    original_norms,
                    delta,
                    scale=edit.strength,
                )
                direct_single_requantization = (
                    not edit.preserve_original_blocks
                    and len(edit.quantization_multipliers) == 1
                )
                if edit.preserve_original_blocks:
                    selected_raw = original_raw.copy()
                    best_error = _block_error(base, target, layout.block_size)
                    choices = np.full(
                        (base.shape[0], blocks_per_row), -1, dtype=np.int16
                    )
                else:
                    selected_raw = (
                        original_raw
                        if direct_single_requantization
                        else np.empty_like(original_raw)
                    )
                    best_error = np.full(
                        (base.shape[0], blocks_per_row), np.inf, dtype=np.float64
                    )
                    choices = np.full(
                        (base.shape[0], blocks_per_row), -2, dtype=np.int16
                    )

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
                realized_candidate: np.ndarray | None = None
                encoded_candidate: np.ndarray | None = None
                for multiplier_index, multiplier in enumerate(
                    edit.quantization_multipliers
                ):
                    candidate = (
                        target
                        if multiplier == 1.0
                        else _edited_from_prepared(
                            prepared_base,
                            original_norms,
                            delta,
                            scale=edit.strength * multiplier,
                        )
                    )
                    encoded_candidate = codec.quantize_rows(candidate, qtype)
                    if encoded_candidate.shape != original_raw.shape:
                        raise RuntimeError(
                            f"same-type codec changed {source_tensor.name} row layout: "
                            f"{encoded_candidate.shape} != {original_raw.shape}"
                        )
                    if direct_single_requantization:
                        choices.fill(multiplier_index)
                        realized_candidate = (
                            codec.dequantize_rows(encoded_candidate, qtype, input_dim)
                            if diagnostics_mode == "full"
                            else None
                        )
                    else:
                        realized_candidate = codec.dequantize_rows(
                            encoded_candidate, qtype, input_dim
                        )
                        candidate_error = _block_error(
                            realized_candidate, target, layout.block_size
                        )
                        threshold = best_error * (
                            1.0 - edit.minimum_block_improvement
                        )
                        better = candidate_error < threshold
                        if bool(better.any()):
                            selected_blocks = selected_raw.reshape(
                                selected_raw.shape[0], blocks_per_row, layout.type_size
                            )
                            candidate_blocks = encoded_candidate.reshape(
                                encoded_candidate.shape[0],
                                blocks_per_row,
                                layout.type_size,
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
                        + (
                            realized_candidate.nbytes
                            if realized_candidate is not None
                            else 0
                        )
                        + selected_raw.nbytes
                        + original_raw.nbytes
                        + best_error.nbytes
                        + choices.nbytes
                        + (projection.nbytes if projection is not None else 0),
                        projection_workspace_bytes,
                    )

                if bool((choices == -2).any()):
                    raise RuntimeError(
                        "no quantization candidate was selected for one or more blocks"
                    )
                if direct_single_requantization:
                    assert encoded_candidate is not None
                    selected_raw = encoded_candidate
                    realized = realized_candidate
                else:
                    realized = (
                        codec.dequantize_rows(selected_raw, qtype, input_dim)
                        if diagnostics_mode == "full"
                        else None
                    )
                assert state.accumulator is not None
                if diagnostics_mode == "full":
                    assert isinstance(state.accumulator, _ErrorAccumulator)
                    assert realized is not None
                    state.accumulator.update(
                        base=base,
                        target=target,
                        realized=realized,
                        original_raw=original_raw,
                        selected_raw=selected_raw,
                        choice_indices=choices,
                        multipliers=edit.quantization_multipliers,
                        type_size=layout.type_size,
                        workspace_bytes=max(workspace_bytes, realized.nbytes),
                        base_norms=original_norms,
                    )
                else:
                    assert isinstance(state.accumulator, _SearchAccumulator)
                    state.accumulator.update(
                        original_raw=original_raw,
                        selected_raw=selected_raw,
                        choice_indices=choices,
                        multipliers=edit.quantization_multipliers,
                        type_size=layout.type_size,
                        workspace_bytes=workspace_bytes,
                    )
                destination_bank[start:stop] = selected_raw
                state.after_digest.update(memoryview(selected_raw))

    before_sha256 = before_digest.hexdigest()
    results: list[dict[str, object]] = []
    for state in states:
        if state.edit.strength == 0:
            results.append(
                _zero_strength_payload(
                    raw,
                    total_blocks=total_blocks,
                    diagnostics_mode=diagnostics_mode,
                )
            )
        else:
            results.append(
                _finish_payload(
                    source_tensor.name,
                    state,
                    before_sha256=before_sha256,
                    diagnostics_mode=diagnostics_mode,
                )
            )
    return results


def _prepare_contexts(
    source: Path,
    tensor_file: Path,
    candidates: Sequence[GGUFStrengthSweepCandidate],
    *,
    fast_search: bool,
) -> tuple[list[_CandidateContext], bytes, str, int, bool]:
    if not 2 <= len(candidates) <= 32:
        raise ValueError("a strength sweep requires between 2 and 32 candidates")
    labels = [candidate.label for candidate in candidates]
    if len(labels) != len(set(labels)):
        raise ValueError("strength sweep labels must be unique")

    contexts: list[_CandidateContext] = []
    output_paths: list[Path] = []
    for spec in candidates:
        plan_path = spec.plan_path.expanduser().resolve(strict=True)
        output_path = spec.output_path.expanduser().resolve(strict=False)
        if output_path == source:
            raise ValueError("source and sweep output GGUF paths must differ")
        if output_path.is_dir():
            raise IsADirectoryError(f"sweep output is a directory: {output_path}")
        if output_path.exists():
            raise FileExistsError(f"refusing to overwrite existing output: {output_path}")
        plan_payload = plan_path.read_bytes()
        plan, edits, row_chunk_size, verify_untouched, arithmetic_mode = _resolve_plan(
            plan_payload
        )
        if (
            not isinstance(plan, GGUFQuantizedAblationPlan)
            or plan.schema_version != "gguf-static-ablation-v3"
            or arithmetic_mode != "chunk-stable-v1"
        ):
            raise ValueError("strength sweeps require gguf-static-ablation-v3 plans")
        contexts.append(
            _CandidateContext(
                spec=spec,
                plan_path=plan_path,
                output_path=output_path,
                plan_sha256=hashlib.sha256(plan_payload).hexdigest(),
                plan=plan,
                edits=edits,
            )
        )
        output_paths.append(output_path)
    if len(output_paths) != len(set(output_paths)):
        raise ValueError("strength sweep output paths must be unique")

    reference = contexts[0]
    reference_invariants = tuple(_edit_invariant(edit) for edit in reference.edits)
    for context in contexts[1:]:
        if (
            context.plan.source_sha256 != reference.plan.source_sha256
            or context.plan.tensor_artifact_sha256
            != reference.plan.tensor_artifact_sha256
            or context.plan.row_chunk_size != reference.plan.row_chunk_size
            or context.plan.verify_untouched_bytes
            != reference.plan.verify_untouched_bytes
            or tuple(_edit_invariant(edit) for edit in context.edits)
            != reference_invariants
        ):
            raise ValueError(
                "strength sweep plans must differ only in per-tensor strength "
                "or its payload-change gate"
            )
    if fast_search and reference.plan.verify_untouched_bytes:
        raise ValueError(
            "fast search requires plans with verify_untouched_bytes=false"
        )
    if fast_search and any(
        gate is not None
        for context in contexts
        for edit in context.edits
        for gate in (
            edit.minimum_delta_cosine,
            edit.maximum_delta_relative_error,
            edit.maximum_row_norm_relative_error,
        )
    ):
        raise ValueError("fast search cannot defer diagnostics required by an edit gate")

    tensor_payload = tensor_file.read_bytes()
    tensor_sha256 = hashlib.sha256(tensor_payload).hexdigest()
    if tensor_sha256 != reference.plan.tensor_artifact_sha256:
        raise RuntimeError(
            f"factor artifact hash mismatch: {tensor_sha256} != "
            f"{reference.plan.tensor_artifact_sha256}"
        )
    return (
        contexts,
        tensor_payload,
        tensor_sha256,
        reference.plan.row_chunk_size,
        reference.plan.verify_untouched_bytes,
    )


def _preflight_source(
    source: Path,
    source_tensors: dict[str, Any],
    source_size: int,
    edits: tuple[_ResolvedEdit, ...],
    factors: dict[str, np.ndarray],
    codec: GGUFQuantizationCodecRegistry,
    contexts: Sequence[_CandidateContext],
    *,
    verify_untouched: bool,
) -> tuple[list[dict[str, object]], tuple[tuple[int, int], ...], _FileHashSnapshot]:
    """Validate tensor arithmetic and bind one stable source snapshot."""

    prepared_rows: list[dict[str, object]] = []
    for edit in edits:
        tensor = source_tensors[edit.tensor_name]
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
                f"target {edit.tensor_name} input dimension {logical_shape[-1]} "
                f"is not divisible by the {qname} block size {layout.block_size}"
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
                    f"projector factors differ for {edit.tensor_name}: "
                    f"A={a.shape}, B={b.shape}"
                )
        else:
            right = factors[edit.right_key]
            if right.shape != (logical_shape[-1], a.shape[1]):
                raise RuntimeError(
                    f"direct right factor mismatch for {edit.tensor_name}: "
                    f"A={a.shape}, right={right.shape}, input={logical_shape[-1]}"
                )
        prepared_rows.append(
            {
                "tensor_name": edit.tensor_name,
                "logical_shape": list(logical_shape),
                "quantization": qname,
                "codec_backend": codec.backend_for(tensor.tensor_type),
                "matrix_count": int(np.prod(logical_shape[:-2])) or 1,
                "rank": int(a.shape[1]),
                "edit_mode": "direct-low-rank" if edit.right_key else "projector",
                "preserve_row_norms": edit.preserve_row_norms,
                "preserve_original_blocks": edit.preserve_original_blocks,
                "quantization_multipliers": list(edit.quantization_multipliers),
                "data_offset": int(tensor.data_offset),
                "quantized_bytes": int(tensor.n_bytes),
            }
        )

    intervals = _validated_intervals(
        source_size,
        [
            (
                int(row["data_offset"]),
                int(row["data_offset"]) + int(row["quantized_bytes"]),
            )
            for row in prepared_rows
        ],
    )
    source_snapshot = _file_and_untouched_sha256(
        source,
        intervals,
        capture_untouched=verify_untouched,
    )
    if source_snapshot.sha256 != contexts[0].plan.source_sha256:
        raise RuntimeError(
            f"source GGUF hash mismatch: {source_snapshot.sha256} != "
            f"{contexts[0].plan.source_sha256}"
        )

    target_bytes = sum(stop - start for start, stop in intervals)
    output_parents = {context.output_path.parent for context in contexts}
    for parent in output_parents:
        parent.mkdir(parents=True, exist_ok=True)
        candidates_in_parent = sum(
            context.output_path.parent == parent for context in contexts
        )
        minimum_free = target_bytes * candidates_in_parent + 64 * 1024 * 1024
        if shutil.disk_usage(parent).free < minimum_free:
            raise RuntimeError(
                f"insufficient free space in {parent}: need at least "
                f"{minimum_free} bytes for edited payloads and transaction headroom"
            )
    return prepared_rows, intervals, source_snapshot


def apply_quantized_gguf_strength_sweep(
    source_path: str | Path,
    tensor_path: str | Path,
    candidates: Sequence[GGUFStrengthSweepCandidate],
    *,
    ggml_library: str | Path | None = None,
    fast_search: bool = False,
) -> dict[str, Any]:
    """Build several exact GGUF strength candidates with shared source work.

    Plans must use schema v3 and have identical edit arithmetic except for the
    strength of each target tensor. The payload-change gate may differ so an
    explicit zero-strength control can share the portfolio. The function never
    overwrites outputs. It validates every temporary artifact before publishing
    any of them and rolls back outputs it created if a later publication loses
    a race.
    """

    started = time.perf_counter()
    GGUFReader, _GGMLQuantizationType, _dequantize, _quantize = _gguf_api()
    source = Path(source_path).expanduser().resolve(strict=True)
    tensor_file = Path(tensor_path).expanduser().resolve(strict=True)
    if not source.is_file() or not tensor_file.is_file():
        raise ValueError("source and tensor artifact must both be files")
    validated_candidates = tuple(
        candidate
        if isinstance(candidate, GGUFStrengthSweepCandidate)
        else GGUFStrengthSweepCandidate.model_validate(candidate)
        for candidate in candidates
    )
    (
        contexts,
        tensor_payload,
        tensor_sha256,
        row_chunk_size,
        verify_untouched,
    ) = _prepare_contexts(
        source,
        tensor_file,
        validated_candidates,
        fast_search=fast_search,
    )
    edits = contexts[0].edits
    factors = _load_factors(tensor_payload, edits)

    output_parents = tuple(dict.fromkeys(context.output_path.parent for context in contexts))
    bound_source = _bind_source_snapshot(source, output_parents)
    source_reader: Any | None = None
    source_tensors: dict[str, Any] = {}
    codec: GGUFQuantizationCodecRegistry | None = None
    try:
        codec = GGUFQuantizationCodecRegistry(ggml_library=ggml_library)
        source_reader = GGUFReader(bound_source.path)
        source_size = bound_source.path.stat().st_size
        _validate_tensor_payload_layout(source_reader, source_size)
        if source_reader.byte_order != "I":
            raise RuntimeError(
                "direct quantized editing refuses opposite-endian GGUF files"
            )
        source_tensors = {tensor.name: tensor for tensor in source_reader.tensors}
        missing = sorted(
            edit.tensor_name for edit in edits if edit.tensor_name not in source_tensors
        )
        if missing:
            raise RuntimeError(f"GGUF is missing target tensors: {missing}")
        prepared_rows, intervals, source_snapshot = _preflight_source(
            bound_source.path,
            source_tensors,
            source_size,
            edits,
            factors,
            codec,
            contexts,
            verify_untouched=verify_untouched,
        )
    except BaseException:
        if codec is not None:
            codec.close()
        del source_tensors
        if source_reader is not None:
            del source_reader
        bound_source.close()
        gc.collect()
        raise
    assert codec is not None

    published: list[tuple[Path, _FileHashSnapshot]] = []
    try:
        target_bytes = sum(stop - start for start, stop in intervals)
        reserved_target_bytes = {
            parent: target_bytes
            * sum(context.output_path.parent == parent for context in contexts)
            + 64 * 1024 * 1024
            for parent in output_parents
        }
        for context_index, context in enumerate(contexts):
            temporary_fd, temporary_name = tempfile.mkstemp(
                prefix=f".{context.output_path.name}.",
                suffix=".tmp",
                dir=context.output_path.parent,
            )
            os.close(temporary_fd)
            context.temporary = Path(temporary_name)
            context.snapshot_copy_mode = _copy_source(
                bound_source.path,
                context.temporary,
                minimum_free_after_copy=reserved_target_bytes[context.output_path.parent],
            )
            if context.snapshot_copy_mode == "copy":
                parent = context.output_path.parent
                remaining_full_copies = sum(
                    later.output_path.parent == parent
                    for later in contexts[context_index + 1 :]
                )
                required_free = (
                    source_snapshot.size_bytes * remaining_full_copies
                    + reserved_target_bytes[parent]
                )
                free = shutil.disk_usage(parent).free
                if free < required_free:
                    raise RuntimeError(
                        "insufficient free space for remaining full GGUF sweep "
                        f"copies in {parent}: need {required_free} bytes, have {free}"
                    )

        if not _stable_source_identity(source_snapshot, bound_source.path):
            raise RuntimeError("source GGUF changed while sweep snapshots were created")

        for context in contexts:
            assert context.temporary is not None
            context.reader = GGUFReader(context.temporary, mode="r+")
            _validate_tensor_payload_layout(context.reader, context.temporary.stat().st_size)
            if context.reader.byte_order != "I":
                raise RuntimeError("sweep snapshot changed byte order unexpectedly")
            destination_tensors = {tensor.name: tensor for tensor in context.reader.tensors}
            for prepared in prepared_rows:
                tensor = destination_tensors[str(prepared["tensor_name"])]
                if (
                    tensor.tensor_type.name != prepared["quantization"]
                    or int(tensor.data_offset) != prepared["data_offset"]
                    or int(tensor.n_bytes) != prepared["quantized_bytes"]
                ):
                    raise RuntimeError(
                        f"sweep snapshot layout changed for {prepared['tensor_name']}"
                    )

        per_candidate_rows: list[list[dict[str, object]]] = [
            [] for _ in contexts
        ]
        shared_edit_seconds = 0.0
        for edit_index, prepared in enumerate(prepared_rows):
            edit_started = time.perf_counter()
            tensor_name = str(prepared["tensor_name"])
            results = _edit_tensor_strength_sweep(
                source_tensors[tensor_name],
                [
                    {tensor.name: tensor for tensor in context.reader.tensors}[tensor_name]
                    for context in contexts
                ],
                [context.edits[edit_index] for context in contexts],
                factors,
                codec,
                row_chunk_size=row_chunk_size,
                diagnostics_mode="search" if fast_search else "full",
            )
            edit_seconds = time.perf_counter() - edit_started
            shared_edit_seconds += edit_seconds
            for candidate_index, (context, payload) in enumerate(
                zip(contexts, results, strict=True)
            ):
                candidate_edit = context.edits[edit_index]
                per_candidate_rows[candidate_index].append(
                    {
                        **prepared,
                        "strength": candidate_edit.strength,
                        **payload,
                        "elapsed_seconds": edit_seconds,
                        "shared_sweep_elapsed_seconds": edit_seconds,
                    }
                )

        if not _stable_source_identity(source_snapshot, bound_source.path):
            raise RuntimeError("source GGUF changed while the strength sweep was running")

        for context, rows in zip(contexts, per_candidate_rows, strict=True):
            context.result_rows = rows
            assert context.reader is not None
            context.reader.data.flush()
            del context.reader
            context.reader = None
            gc.collect()
            assert context.temporary is not None
            descriptor = os.open(context.temporary, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)

        expected_payloads = {
            (
                int(row["data_offset"]),
                int(row["data_offset"]) + int(row["quantized_bytes"]),
            ): index
            for index, row in enumerate(prepared_rows)
        }
        for context in contexts:
            assert context.temporary is not None and context.result_rows is not None
            final_hashes = _file_and_untouched_sha256(
                context.temporary,
                intervals,
                capture_interval_hashes=True,
                capture_untouched=verify_untouched,
            )
            for interval, payload_sha256 in zip(
                intervals, final_hashes.interval_sha256, strict=True
            ):
                edit_index = expected_payloads[interval]
                expected_sha256 = str(
                    context.result_rows[edit_index]["after_payload_sha256"]
                )
                if payload_sha256 != expected_sha256:
                    raise RuntimeError(
                        "post-write GGUF validation failed for "
                        f"{context.result_rows[edit_index]['tensor_name']} "
                        f"in candidate {context.spec.label}"
                    )
            if verify_untouched and (
                final_hashes.untouched_sha256
                != source_snapshot.untouched_sha256
            ):
                raise RuntimeError(
                    f"GGUF bytes changed outside declared tensor payloads for "
                    f"candidate {context.spec.label}"
                )
            context.final_hashes = final_hashes

        for context in contexts:
            assert context.temporary is not None and context.final_hashes is not None
            try:
                _publish_output(
                    context.temporary,
                    context.output_path,
                    force=False,
                    expected=context.final_hashes,
                )
            except BaseException:
                # _publish_output may have installed the hard link before a
                # post-link identity check fails. Remove only the exact inode
                # created by this transaction; never touch a racing file.
                try:
                    current = context.output_path.stat()
                    if (
                        current.st_dev == context.final_hashes.device
                        and current.st_ino == context.final_hashes.inode
                    ):
                        context.output_path.unlink()
                except FileNotFoundError:
                    pass
                raise
            published.append((context.output_path, context.final_hashes))

        total_elapsed = time.perf_counter() - started
        candidate_reports = []
        for context in contexts:
            assert context.result_rows is not None and context.final_hashes is not None
            candidate_reports.append(
                {
                    "label": context.spec.label,
                    "merge_report": {
                        "schema_version": "gguf-quantized-static-merge-report-v3",
                        "source": {
                            "path": str(source),
                            "sha256": source_snapshot.sha256,
                            "size_bytes": source_snapshot.size_bytes,
                        },
                        "plan": {
                            "path": str(context.plan_path),
                            "sha256": context.plan_sha256,
                            "schema_version": context.plan.schema_version,
                        },
                        "tensor_artifact": {
                            "path": str(tensor_file),
                            "sha256": tensor_sha256,
                        },
                        "codec": codec.provenance(),
                        "row_chunk_size": row_chunk_size,
                        "arithmetic_mode": "chunk-stable-v1",
                        "verify_untouched_bytes": verify_untouched,
                        "search_only": not verify_untouched,
                        "diagnostics_mode": "search" if fast_search else "full",
                        "execution_mode": "shared-strength-sweep-v1",
                        "source_binding_mode": bound_source.mode,
                        "snapshot_copy_mode": context.snapshot_copy_mode,
                        "dry_run": False,
                        "edits": context.result_rows,
                        **(
                            {
                                "untouched_bytes_sha256": (
                                    source_snapshot.untouched_sha256
                                )
                            }
                            if verify_untouched
                            else {}
                        ),
                        "untouched_bytes_verified": verify_untouched,
                        "output": {
                            "path": str(context.output_path),
                            "sha256": context.final_hashes.sha256,
                            "size_bytes": context.final_hashes.size_bytes,
                        },
                        "elapsed_seconds": total_elapsed,
                    },
                }
            )
        return {
            "schema_version": "gguf-quantized-strength-sweep-report-v1",
            "source": {
                "path": str(source),
                "sha256": source_snapshot.sha256,
                "size_bytes": source_snapshot.size_bytes,
            },
            "tensor_artifact": {
                "path": str(tensor_file),
                "sha256": tensor_sha256,
            },
            "candidate_count": len(contexts),
            "source_binding_mode": bound_source.mode,
            "snapshot_copy_modes": [
                context.snapshot_copy_mode for context in contexts
            ],
            "shared_source_dequantization": any(
                edit.strength > 0 for context in contexts for edit in context.edits
            ),
            "shared_projector_reduction": any(
                edit.right_key is None and edit.strength > 0
                for context in contexts
                for edit in context.edits
            ),
            "shared_edit_seconds": shared_edit_seconds,
            "candidates": candidate_reports,
            "elapsed_seconds": total_elapsed,
        }
    except BaseException:
        for output_path, expected in reversed(published):
            try:
                current = output_path.stat()
                if current.st_dev == expected.device and current.st_ino == expected.inode:
                    output_path.unlink()
            except FileNotFoundError:
                pass
        raise
    finally:
        for context in contexts:
            if context.reader is not None:
                try:
                    del context.reader
                except Exception:
                    pass
                context.reader = None
            if context.temporary is not None and context.temporary.exists():
                context.temporary.unlink()
        codec.close()
        del source_tensors
        if source_reader is not None:
            del source_reader
        bound_source.close()
        gc.collect()
