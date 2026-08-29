"""Bounded-memory static ablation directly inside Q8_0 GGUF files.

The input GGUF is copied byte-for-byte and only explicitly declared tensor
payloads are replaced. Each selected matrix is dequantized independently,
edited in FP32, requantized to Q8_0, and written back at the same offset. GGUF
metadata and every non-selected tensor therefore remain untouched.
"""

from __future__ import annotations

import gc
import hashlib
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from safetensors import safe_open

from heretic_nx.hashing import canonical_json, sha256_file


SHA256_PATTERN = r"^[0-9a-f]{64}$"


class GGUFQ8TensorEdit(BaseModel):
    """One low-rank output-space ablation applied to a GGUF matrix."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    tensor_name: str = Field(min_length=1)
    a_key: str = Field(min_length=1)
    b_key: str | None = None
    right_key: str | None = None
    strength: float = Field(ge=0)
    preserve_row_norms: bool = True

    @model_validator(mode="after")
    def validate_factor_mode(self) -> "GGUFQ8TensorEdit":
        if self.right_key is not None and self.b_key is not None:
            raise ValueError("right_key and b_key are mutually exclusive")
        if self.right_key is not None and self.preserve_row_norms:
            raise ValueError("direct right-factor edits cannot preserve row norms")
        return self

    @property
    def resolved_b_key(self) -> str:
        return self.b_key or self.a_key


class GGUFQ8AblationPlan(BaseModel):
    """Hash-bound plan for an in-container Q8_0 static merge."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["gguf-q8-ablation-v1"] = "gguf-q8-ablation-v1"
    source_sha256: str = Field(pattern=SHA256_PATTERN)
    tensor_artifact_sha256: str = Field(pattern=SHA256_PATTERN)
    edits: tuple[GGUFQ8TensorEdit, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_targets(self) -> "GGUFQ8AblationPlan":
        targets = [edit.tensor_name for edit in self.edits]
        if len(targets) != len(set(targets)):
            raise ValueError("each GGUF tensor may be edited at most once")
        return self

    @classmethod
    def read(cls, path: str | Path) -> "GGUFQ8AblationPlan":
        return cls.model_validate_json(Path(path).read_bytes())

    def write(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(canonical_json(self.model_dump()) + b"\n")


def _gguf_api() -> tuple[Any, Any, Any, Any]:
    try:
        from gguf import GGMLQuantizationType, GGUFReader
        from gguf.quants import dequantize, quantize
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "direct GGUF editing requires the 'gguf' extra: "
            "python -m pip install -e '.[gguf]'"
        ) from error
    return GGUFReader, GGMLQuantizationType, dequantize, quantize


def _matrix_factor(value: np.ndarray, *, key: str) -> np.ndarray:
    factor = np.asarray(value, dtype=np.float32)
    if factor.ndim == 1:
        factor = factor[:, None]
    if factor.ndim != 2 or factor.shape[0] == 0 or factor.shape[1] == 0:
        raise ValueError(f"factor {key!r} must be a non-empty vector or matrix")
    if not np.isfinite(factor).all():
        raise ValueError(f"factor {key!r} contains a non-finite value")
    return np.ascontiguousarray(factor)


def _load_factors(
    tensor_path: Path,
    plan: GGUFQ8AblationPlan,
) -> dict[str, np.ndarray]:
    required = {
        key
        for edit in plan.edits
        for key in (
            edit.a_key,
            edit.right_key if edit.right_key is not None else edit.resolved_b_key,
        )
    }
    factors: dict[str, np.ndarray] = {}
    with safe_open(tensor_path, framework="numpy") as artifact:
        available = set(artifact.keys())
        missing = sorted(required - available)
        if missing:
            raise RuntimeError(f"tensor artifact is missing factors: {missing}")
        for key in sorted(required):
            factors[key] = _matrix_factor(artifact.get_tensor(key), key=key)
    return factors


def _sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(memoryview(np.ascontiguousarray(value))).hexdigest()


def _copy_source(source: Path, target: Path) -> None:
    """Clone on APFS when available, with a portable byte-copy fallback."""

    if sys.platform == "darwin" and Path("/bin/cp").is_file():
        try:
            subprocess.run(
                ("/bin/cp", "-c", str(source), str(target)),
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return
        except subprocess.CalledProcessError:
            pass
    shutil.copyfile(source, target)


def _norm_preserving_ablation(
    weight: np.ndarray,
    a: np.ndarray,
    b: np.ndarray | None,
    *,
    right: np.ndarray | None = None,
    strength: float,
    preserve_row_norms: bool,
) -> np.ndarray:
    """Apply an output projector or a direct low-rank delta to matrix banks."""

    matrix = np.ascontiguousarray(weight, dtype=np.float32)
    if matrix.ndim < 2:
        raise ValueError("Q8 ablation targets must dequantize to matrices")
    output_dim = matrix.shape[-2]
    if a.shape[0] != output_dim:
        raise ValueError(
            "factor output dimension does not match the target matrix: "
            f"A={a.shape}, weight={matrix.shape}"
        )
    if right is None:
        if b is None or b.shape[0] != output_dim:
            raise ValueError(
                "projector factor dimension does not match the target matrix: "
                f"B={None if b is None else b.shape}, weight={matrix.shape}"
            )
        if a.shape[1] != b.shape[1]:
            raise ValueError(f"factor ranks differ: A={a.shape}, B={b.shape}")
    else:
        if preserve_row_norms:
            raise ValueError("direct right-factor edits cannot preserve row norms")
        if right.shape != (matrix.shape[-1], a.shape[1]):
            raise ValueError(
                "direct right factor must be [input_dim, rank]: "
                f"A={a.shape}, right={right.shape}, weight={matrix.shape}"
            )
    if not math.isfinite(strength) or strength < 0:
        raise ValueError("strength must be finite and non-negative")
    if not np.isfinite(matrix).all():
        raise ValueError("dequantized target contains a non-finite value")

    banks = matrix.reshape(-1, matrix.shape[-2], matrix.shape[-1])
    for bank in banks:
        original_norms: np.ndarray | None = None
        if preserve_row_norms:
            original_norms = np.linalg.norm(bank, axis=1, keepdims=True)
            normalized = np.zeros_like(bank)
            np.divide(bank, original_norms, out=normalized, where=original_norms > 0)
            bank[...] = normalized

        if right is None:
            assert b is not None
            projection = b.T @ bank
            bank -= np.float32(strength) * (a @ projection)
        else:
            bank -= np.float32(strength) * (a @ right.T)

        if original_norms is not None:
            edited_norms = np.linalg.norm(bank, axis=1, keepdims=True)
            collapsed = (original_norms[:, 0] > 1e-7) & (
                edited_norms[:, 0] <= 1e-7
            )
            if bool(collapsed.any()):
                raise RuntimeError("Q8 ablation collapsed a non-zero output row")
            np.divide(bank, edited_norms, out=bank, where=edited_norms > 0)
            bank *= original_norms

    if not np.isfinite(matrix).all():
        raise RuntimeError("Q8 ablation produced a non-finite matrix")
    return np.ascontiguousarray(matrix, dtype=np.float32)


def inspect_q8_gguf(path: str | Path) -> dict[str, Any]:
    """Return edit-relevant Q8_0 tensor metadata without dequantizing weights."""

    GGUFReader, GGMLQuantizationType, _dequantize, _quantize = _gguf_api()
    source = Path(path).expanduser().resolve(strict=True)
    reader = GGUFReader(source)
    q8 = [
        {
            "name": tensor.name,
            "shape": [int(value) for value in reversed(tensor.shape.tolist())],
            "quantized_bytes": int(tensor.n_bytes),
            "data_offset": int(tensor.data_offset),
        }
        for tensor in reader.tensors
        if tensor.tensor_type == GGMLQuantizationType.Q8_0
    ]
    return {
        "schema_version": "gguf-q8-inspection-v1",
        "path": str(source),
        "size_bytes": source.stat().st_size,
        "tensor_count": len(reader.tensors),
        "q8_0_tensor_count": len(q8),
        "q8_0_tensors": q8,
    }


def apply_q8_gguf_ablation(
    source_path: str | Path,
    output_path: str | Path | None,
    plan_path: str | Path,
    tensor_path: str | Path,
    *,
    dry_run: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    """Apply a hash-bound plan directly to Q8_0 tensor payloads in a GGUF."""

    GGUFReader, GGMLQuantizationType, dequantize, quantize = _gguf_api()
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

    plan = GGUFQ8AblationPlan.read(plan_file)
    source_sha256 = sha256_file(source)
    if source_sha256 != plan.source_sha256:
        raise RuntimeError(
            f"source GGUF hash mismatch: {source_sha256} != {plan.source_sha256}"
        )
    tensor_sha256 = sha256_file(tensor_file)
    if tensor_sha256 != plan.tensor_artifact_sha256:
        raise RuntimeError(
            "factor artifact hash mismatch: "
            f"{tensor_sha256} != {plan.tensor_artifact_sha256}"
        )
    factors = _load_factors(tensor_file, plan)

    reader = GGUFReader(source)
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    missing_targets = sorted(
        edit.tensor_name for edit in plan.edits if edit.tensor_name not in tensors
    )
    if missing_targets:
        raise RuntimeError(f"GGUF is missing target tensors: {missing_targets}")

    prepared = []
    for edit in plan.edits:
        tensor = tensors[edit.tensor_name]
        if tensor.tensor_type != GGMLQuantizationType.Q8_0:
            raise RuntimeError(
                f"target {edit.tensor_name} is {tensor.tensor_type.name}, not Q8_0"
            )
        a = factors[edit.a_key]
        right = factors[edit.right_key] if edit.right_key is not None else None
        b = factors[edit.resolved_b_key] if right is None else None
        logical_shape = tuple(int(value) for value in reversed(tensor.shape.tolist()))
        if len(logical_shape) < 2:
            raise RuntimeError(f"target {edit.tensor_name} is not a matrix or expert bank")
        output_dim = logical_shape[-2]
        if a.shape[0] != output_dim:
            raise RuntimeError(
                f"factor dimension mismatch for {edit.tensor_name}: "
                f"expected output {output_dim}, got A={a.shape[0]}"
            )
        if right is None:
            assert b is not None
            if b.shape[0] != output_dim or a.shape[1] != b.shape[1]:
                raise RuntimeError(
                    f"projector factor mismatch for {edit.tensor_name}: "
                    f"A={a.shape}, B={b.shape}, output={output_dim}"
                )
        elif right.shape != (logical_shape[-1], a.shape[1]):
            raise RuntimeError(
                f"direct right-factor mismatch for {edit.tensor_name}: "
                f"A={a.shape}, right={right.shape}, input={logical_shape[-1]}"
            )
        prepared.append(
            {
                "tensor_name": edit.tensor_name,
                "logical_shape": list(logical_shape),
                "quantization": tensor.tensor_type.name,
                "matrix_count": int(np.prod(logical_shape[:-2])) or 1,
                "rank": int(a.shape[1]),
                "edit_mode": "direct-low-rank" if right is not None else "projector",
                "strength": edit.strength,
                "preserve_row_norms": edit.preserve_row_norms,
                "data_offset": int(tensor.data_offset),
                "quantized_bytes": int(tensor.n_bytes),
            }
        )

    report: dict[str, Any] = {
        "schema_version": "gguf-q8-static-merge-report-v1",
        "source": {
            "path": str(source),
            "sha256": source_sha256,
            "size_bytes": source.stat().st_size,
        },
        "plan": {"path": str(plan_file), "sha256": sha256_file(plan_file)},
        "tensor_artifact": {"path": str(tensor_file), "sha256": tensor_sha256},
        "dry_run": dry_run,
        "edits": prepared,
    }
    if dry_run:
        return report

    assert output is not None
    output.parent.mkdir(parents=True, exist_ok=True)
    required_bytes = source.stat().st_size + 64 * 1024 * 1024
    free_bytes = shutil.disk_usage(output.parent).free
    if free_bytes < required_bytes:
        raise RuntimeError(
            f"insufficient free space: need at least {required_bytes} bytes, "
            f"have {free_bytes}"
        )

    temporary_fd, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    os.close(temporary_fd)
    temporary = Path(temporary_name)
    try:
        _copy_source(source, temporary)
        writable = GGUFReader(temporary, mode="r+")
        writable_tensors = {tensor.name: tensor for tensor in writable.tensors}
        result_rows = []
        for edit in plan.edits:
            tensor = writable_tensors[edit.tensor_name]
            before_sha256 = _sha256_array(tensor.data)
            weight = dequantize(tensor.data, tensor.tensor_type)
            edited = _norm_preserving_ablation(
                weight,
                factors[edit.a_key],
                (
                    factors[edit.resolved_b_key]
                    if edit.right_key is None
                    else None
                ),
                right=(
                    factors[edit.right_key]
                    if edit.right_key is not None
                    else None
                ),
                strength=edit.strength,
                preserve_row_norms=edit.preserve_row_norms,
            )
            requantized = quantize(edited, tensor.tensor_type)
            if requantized.shape != tensor.data.shape or requantized.dtype != tensor.data.dtype:
                raise RuntimeError(
                    f"requantized payload layout changed for {edit.tensor_name}: "
                    f"{requantized.shape}/{requantized.dtype} != "
                    f"{tensor.data.shape}/{tensor.data.dtype}"
                )
            tensor.data[...] = requantized
            after_sha256 = _sha256_array(tensor.data)
            result_rows.append(
                {
                    **next(row for row in prepared if row["tensor_name"] == edit.tensor_name),
                    "before_payload_sha256": before_sha256,
                    "after_payload_sha256": after_sha256,
                    "payload_changed": before_sha256 != after_sha256,
                }
            )
            del weight, edited, requantized
            gc.collect()
        writable.data.flush()
        del tensor
        del writable_tensors
        del writable
        gc.collect()

        os.replace(temporary, output)
        report["dry_run"] = False
        report["edits"] = result_rows
        report["output"] = {
            "path": str(output),
            "sha256": sha256_file(output),
            "size_bytes": output.stat().st_size,
        }
        return report
    finally:
        if temporary.exists():
            temporary.unlink()
