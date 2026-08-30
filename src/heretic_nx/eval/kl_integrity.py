"""Integrity checks for persisted first-token KL inputs.

KL reports are release evidence.  A preallocated matrix has its final shape before
collection finishes, so shape alone cannot distinguish a complete run from an
interrupted one.  This module keeps the persistence checks separate from the GGUF
runtime and makes every comparator validate both the progress sidecar and the data.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np


LOG_PROBABILITY_MASS_ATOL = 1e-3
KL_NEGATIVE_ATOL = 1e-8
_SHA256 = re.compile(r"[0-9a-f]{64}")


def default_progress_path(data_path: Path) -> Path:
    """Return the conventional progress sidecar path for an array or raw file."""

    return data_path.with_suffix(".progress.json")


def _read_progress(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise RuntimeError(f"missing KL progress manifest: {path}") from error
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid KL progress manifest: {path}: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"KL progress manifest is not an object: {path}")
    return value


def _require_exact_integer(
    progress: Mapping[str, Any], key: str, expected: int, path: Path
) -> None:
    value = progress.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value != expected:
        raise RuntimeError(
            f"invalid KL progress {key} in {path}: {value!r} != {expected}"
        )


def _require_sha256(progress: Mapping[str, Any], key: str, path: Path) -> None:
    value = progress.get(key)
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise RuntimeError(f"invalid KL progress {key} in {path}: {value!r}")


def load_completed_progress(
    path: Path,
    *,
    schema_version: str,
    count: int,
    vocab_size: int,
    label: str | None = None,
    required_values: Mapping[str, Any] | None = None,
    require_runtime_attestation: bool = True,
) -> dict[str, Any]:
    """Load and validate a complete KL progress manifest.

    Extra fields remain allowed so future collectors can extend a schema without
    invalidating already compatible comparators.
    """

    progress = _read_progress(path)
    expected: dict[str, Any] = {"schema_version": schema_version}
    if label is not None:
        expected["label"] = label
    if required_values is not None:
        expected.update(required_values)
    for key, value in expected.items():
        if progress.get(key) != value:
            raise RuntimeError(
                f"invalid KL progress {key} in {path}: "
                f"{progress.get(key)!r} != {value!r}"
            )

    _require_exact_integer(progress, "count", count, path)
    _require_exact_integer(progress, "vocab_size", vocab_size, path)
    _require_exact_integer(progress, "completed", count, path)
    _require_sha256(progress, "prompt_tokens_sha256", path)
    _require_sha256(progress, "artifact_sha256", path)

    run_label = progress.get("label")
    if not isinstance(run_label, str) or not run_label.strip():
        raise RuntimeError(f"invalid KL progress label in {path}: {run_label!r}")
    model = progress.get("model")
    if not isinstance(model, str) or not model.strip():
        raise RuntimeError(f"invalid KL progress model in {path}: {model!r}")
    seconds = progress.get("seconds")
    if (
        isinstance(seconds, bool)
        or not isinstance(seconds, (int, float))
        or not math.isfinite(float(seconds))
        or float(seconds) < 0.0
    ):
        raise RuntimeError(f"invalid KL progress seconds in {path}: {seconds!r}")
    if require_runtime_attestation:
        runtime = progress.get("runtime_model")
        if not isinstance(runtime, dict):
            raise RuntimeError(f"missing runtime model attestation in {path}")
        for key in (
            "endpoint",
            "model_alias",
            "model_ftype",
            "model_path",
            "build_info",
        ):
            value = runtime.get(key)
            if not isinstance(value, str) or not value.strip():
                raise RuntimeError(
                    f"invalid runtime model attestation {key} in {path}: {value!r}"
                )
        runtime_sha256 = runtime.get("artifact_sha256")
        if (
            not isinstance(runtime_sha256, str)
            or _SHA256.fullmatch(runtime_sha256) is None
            or runtime_sha256 != progress["artifact_sha256"]
        ):
            raise RuntimeError(f"runtime artifact hash mismatch in {path}")
        size = runtime.get("artifact_size_bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size < 1:
            raise RuntimeError(f"invalid runtime artifact size in {path}: {size!r}")
        if runtime["model_alias"] != progress["model"]:
            raise RuntimeError(f"runtime model alias mismatch in {path}")
    return progress


def require_matching_prompt_set(
    base: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    base_path: Path,
    candidate_path: Path,
) -> None:
    """Reject comparisons collected from different ordered prompt sets."""

    if base["prompt_tokens_sha256"] != candidate["prompt_tokens_sha256"]:
        raise RuntimeError(
            "base and candidate KL progress manifests use different prompt sets: "
            f"{base_path} != {candidate_path}"
        )


def require_distinct_artifacts(
    base: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> None:
    """Reject a nominal comparison that actually evaluates identical bytes."""

    if base["artifact_sha256"] == candidate["artifact_sha256"]:
        raise RuntimeError("base and candidate KL artifacts have identical SHA-256")


def require_matching_runtime_protocol(
    base: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> None:
    """Require matched llama.cpp builds and same-quantization runtime arms."""

    base_runtime = base["runtime_model"]
    candidate_runtime = candidate["runtime_model"]
    for key in ("build_info", "model_ftype"):
        if base_runtime[key] != candidate_runtime[key]:
            raise RuntimeError(
                f"base and candidate KL runtimes differ for {key}: "
                f"{base_runtime[key]!r} != {candidate_runtime[key]!r}"
            )


def _log_probability_mass(row: np.ndarray) -> float:
    maximum = float(np.max(row))
    return maximum + math.log(float(np.exp(row - maximum).sum()))


def _log_probability_mass_with_workspace(
    row: np.ndarray, workspace: np.ndarray
) -> float:
    """Compute stable log-sum-exp in a reusable float64 workspace.

    Reusing the output avoids allocating both a float64 conversion and a second
    shifted exponential array for every vocabulary row.  ``dtype`` is explicit
    so float32 inputs are promoted before subtraction, preserving the previous
    float64 arithmetic exactly.
    """

    maximum = float(np.max(row))
    np.subtract(row, maximum, out=workspace, dtype=np.float64)
    np.exp(workspace, out=workspace)
    return maximum + math.log(float(np.sum(workspace, dtype=np.float64)))


def first_token_kl(log_p: np.ndarray, log_q: np.ndarray) -> float:
    """Compute normalized full-vocabulary KL in float64 with fail-closed checks."""

    base = np.asarray(log_p, dtype=np.float64)
    candidate = np.asarray(log_q, dtype=np.float64)
    if base.ndim != 1 or candidate.shape != base.shape or base.size < 2:
        raise ValueError("KL log-probability rows must be aligned vectors")
    if not np.isfinite(base).all() or not np.isfinite(candidate).all():
        raise RuntimeError("KL log-probability rows must be finite")
    base = base - _log_probability_mass(base)
    candidate = candidate - _log_probability_mass(candidate)
    probability = np.exp(base)
    value = float(np.sum(probability * (base - candidate), dtype=np.float64))
    if not math.isfinite(value) or value < -KL_NEGATIVE_ATOL:
        raise RuntimeError(f"invalid KL divergence: {value}")
    return max(0.0, value)


def validate_log_probability_matrix(
    matrix: np.ndarray,
    *,
    path: Path,
    shape: tuple[int, int],
    mass_atol: float = LOG_PROBABILITY_MASS_ATOL,
) -> None:
    """Validate shape, representation, finiteness, and normalization row by row."""

    if matrix.shape != shape:
        raise RuntimeError(
            f"invalid KL matrix shape for {path}: {matrix.shape} != {shape}"
        )
    if matrix.dtype != np.dtype(np.float32):
        raise RuntimeError(
            f"invalid KL matrix dtype for {path}: {matrix.dtype} != float32"
        )
    workspace = np.empty(shape[1], dtype=np.float64)
    for index in range(shape[0]):
        row = matrix[index]
        if not np.isfinite(row).all():
            raise RuntimeError(f"non-finite KL log probabilities in {path} row {index}")
        log_mass = _log_probability_mass_with_workspace(row, workspace)
        if not math.isfinite(log_mass) or abs(log_mass) > mass_atol:
            raise RuntimeError(
                f"invalid KL log-probability distribution in {path} row {index}: "
                f"log mass {log_mass}"
            )


def load_completed_log_probabilities(
    array_path: Path,
    progress_path: Path,
    *,
    schema_version: str,
    label: str,
    count: int,
    vocab_size: int,
    required_values: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load one fully collected and valid log-probability matrix."""

    progress = load_completed_progress(
        progress_path,
        schema_version=schema_version,
        count=count,
        vocab_size=vocab_size,
        label=label,
        required_values=required_values,
    )
    try:
        matrix = np.load(array_path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as error:
        raise RuntimeError(f"invalid KL matrix file: {array_path}: {error}") from error
    validate_log_probability_matrix(
        matrix,
        path=array_path,
        shape=(count, vocab_size),
    )
    return matrix, progress


def load_completed_raw_logits(
    data_path: Path,
    progress_path: Path,
    *,
    schema_version: str,
    count: int,
    vocab_size: int,
) -> tuple[np.memmap, dict[str, Any]]:
    """Load raw float32 logits tied to a complete, content-addressed manifest."""

    progress = load_completed_progress(
        progress_path,
        schema_version=schema_version,
        count=count,
        vocab_size=vocab_size,
    )
    _require_sha256(progress, "data_sha256", progress_path)
    expected_bytes = count * vocab_size * np.dtype(np.float32).itemsize
    try:
        actual_bytes = data_path.stat().st_size
    except FileNotFoundError as error:
        raise RuntimeError(f"missing raw logits file: {data_path}") from error
    if actual_bytes != expected_bytes:
        raise RuntimeError(
            f"invalid raw logits size for {data_path}: "
            f"{actual_bytes} != {expected_bytes}"
        )
    row_nbytes = vocab_size * np.dtype(np.float32).itemsize
    row_buffer = bytearray(row_nbytes)
    row_view = memoryview(row_buffer)
    workspace = np.empty(vocab_size, dtype=np.float64)
    digest = hashlib.sha256()
    with data_path.open("rb") as stream:
        opened_stat = os.fstat(stream.fileno())
        if opened_stat.st_size != expected_bytes:
            raise RuntimeError(
                f"raw logits changed before validation: {data_path}"
            )
        for index in range(count):
            offset = 0
            while offset < row_nbytes:
                size = stream.readinto(row_view[offset:])
                if size is None or size == 0:
                    raise RuntimeError(
                        f"raw logits ended early in {data_path} row {index}"
                    )
                offset += size
            digest.update(row_view)
            row = np.frombuffer(row_view, dtype=np.float32, count=vocab_size)
            if not np.isfinite(row).all():
                raise RuntimeError(
                    f"non-finite raw logits in {data_path} row {index}"
                )
            maximum = float(np.max(row))
            if maximum == float(np.min(row)):
                raise RuntimeError(
                    f"degenerate raw logits in {data_path} row {index}"
                )

            # Normalize once.  The previous implementation exponentiated the
            # entire vocabulary a second time merely to re-check the mass.  The
            # shifted exponential sum already contains exactly the information
            # needed for that check.  For extreme magnitudes near the tolerance,
            # retain the original calculation as a fail-closed fallback.
            np.subtract(row, maximum, out=workspace, dtype=np.float64)
            np.exp(workspace, out=workspace)
            shifted_log_mass = math.log(
                float(np.sum(workspace, dtype=np.float64))
            )
            normalizer = maximum + shifted_log_mass
            log_mass = (maximum - normalizer) + shifted_log_mass
            spacing = abs(float(np.spacing(maximum)))
            if (
                spacing > LOG_PROBABILITY_MASS_ATOL / 16.0
                or abs(log_mass) > LOG_PROBABILITY_MASS_ATOL / 2.0
            ):
                np.copyto(workspace, row, casting="unsafe")
                normalized = workspace - _log_probability_mass(workspace)
                log_mass = _log_probability_mass(normalized)
            if (
                not math.isfinite(log_mass)
                or abs(log_mass) > LOG_PROBABILITY_MASS_ATOL
            ):
                raise RuntimeError(
                    f"invalid normalized log-probability distribution in "
                    f"{data_path} row {index}"
                )

        closed_stat = os.fstat(stream.fileno())
        if (
            opened_stat.st_size != closed_stat.st_size
            or opened_stat.st_mtime_ns != closed_stat.st_mtime_ns
            or opened_stat.st_ctime_ns != closed_stat.st_ctime_ns
        ):
            raise RuntimeError(f"raw logits changed during validation: {data_path}")
        actual_sha256 = digest.hexdigest()
        if progress["data_sha256"] != actual_sha256:
            raise RuntimeError(
                f"raw logits hash does not match {progress_path}: "
                f"{actual_sha256} != {progress['data_sha256']}"
            )
        # Bind the returned mapping to the file descriptor whose bytes were
        # hashed, so a concurrent pathname replacement cannot swap the artifact.
        matrix = np.memmap(
            stream,
            mode="r",
            dtype=np.float32,
            shape=(count, vocab_size),
        )
    return matrix, progress
