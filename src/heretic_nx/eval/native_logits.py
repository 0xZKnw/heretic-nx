"""Fail-closed orchestration for native llama.cpp raw-logit collection.

The native collector avoids materializing a full-vocabulary JSON response for
every prompt.  This module makes that fast path suitable for evaluation
evidence: the exact model inode, tokenizer assets, ordered token rows,
collector executable, dynamic runtime bundle, output bytes, and numerical
shape are all validated and content-addressed before publication.
"""

from __future__ import annotations

from dataclasses import dataclass
import errno
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

from heretic_nx.eval.kl_integrity import (
    default_progress_path,
    load_completed_raw_logits,
)
from heretic_nx.hashing import canonical_json, sha256_bytes, sha256_file, sha256_json


NATIVE_COLLECTOR_SCHEMA = "llama-raw-logits-native-v2"
TOKENIZER_ATTESTATION_SCHEMA = "heretic-nx-tokenizer-assets-v1"
RUNTIME_PROTOCOL_SCHEMA = "heretic-nx-native-logits-runtime-v1"
MODEL_ARTIFACT_SCHEMA = "heretic-nx-gguf-artifact-bundle-v1"
RAW_LOGIT_FORMAT = "float32-little-endian"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_TOKENIZER_NAMES = {
    "added_tokens.json",
    "chat_template.jinja",
    "config.json",
    "merges.txt",
    "sentencepiece.bpe.model",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "vocab.json",
    "vocab.txt",
}
_TOKENIZER_PREFIXES = (
    "chat_template",
    "configuration_",
    "sentencepiece",
    "tokenization_",
    "tokenizer.",
    "vocab.",
)
_RUNTIME_LIBRARY_SUFFIXES = (".dylib", ".dll", ".metallib", ".so")
_INT32_MAX = 2_147_483_647
_INT32_MIN = -2_147_483_648
_UINT32_MAX = 4_294_967_295
_GGUF_SPLIT_NAME = re.compile(
    r"^(?P<prefix>.+)-(?P<part>[0-9]{5})-of-(?P<total>[0-9]{5})[.]gguf$"
)


@dataclass(frozen=True)
class NativeRawLogitsResult:
    """One complete immutable raw-logit artifact and its progress manifest."""

    data_path: Path
    progress_path: Path
    progress: dict[str, Any]
    reused: bool


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _require_positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _resolve_file(path: str | Path, name: str) -> Path:
    try:
        resolved = Path(path).expanduser().resolve(strict=True)
    except FileNotFoundError as error:
        raise ValueError(f"missing {name}: {path}") from error
    if not resolved.is_file():
        raise ValueError(f"{name} is not a regular file: {resolved}")
    return resolved


def _is_tokenizer_asset(path: Path) -> bool:
    name = path.name.lower()
    return name in _TOKENIZER_NAMES or name.startswith(_TOKENIZER_PREFIXES)


def _tokenizer_asset_paths(root: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file()
            and not any(part.startswith(".") for part in path.relative_to(root).parts)
            and _is_tokenizer_asset(path)
        ),
        key=lambda path: path.relative_to(root).as_posix(),
    )


def _tokenizer_relative_paths(root: Path) -> list[str]:
    return [path.relative_to(root).as_posix() for path in _tokenizer_asset_paths(root)]


def _stable_tokenizer_file_record(root: Path, path: Path) -> dict[str, Any]:
    try:
        before = path.stat()
        digest = sha256_file(path)
        after = path.stat()
    except FileNotFoundError as error:
        raise RuntimeError(f"tokenizer asset changed while hashing: {path}") from error
    before_signature = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    after_signature = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if before_signature != after_signature:
        raise RuntimeError(f"tokenizer asset changed while hashing: {path}")
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "size_bytes": after.st_size,
        "sha256": digest,
    }


def attest_tokenizer_assets(
    directory: str | Path,
    *,
    vocab_size: int,
    tokenizer_class: str,
) -> dict[str, Any]:
    """Hash the local assets that define tokenization and prompt rendering."""

    size = _require_positive_integer(vocab_size, "tokenizer vocab_size")
    if not isinstance(tokenizer_class, str) or not tokenizer_class.strip():
        raise ValueError("tokenizer_class must be a non-empty string")
    try:
        root = Path(directory).expanduser().resolve(strict=True)
    except FileNotFoundError as error:
        raise ValueError(f"missing tokenizer directory: {directory}") from error
    if not root.is_dir():
        raise ValueError(f"tokenizer path is not a directory: {root}")
    paths = _tokenizer_asset_paths(root)
    if not paths:
        raise RuntimeError(f"no tokenizer assets discovered under {root}")
    expected_paths = [path.relative_to(root).as_posix() for path in paths]
    files = [_stable_tokenizer_file_record(root, path) for path in paths]
    if _tokenizer_relative_paths(root) != expected_paths:
        raise RuntimeError("tokenizer asset set changed while hashing")
    identity = {
        "schema_version": TOKENIZER_ATTESTATION_SCHEMA,
        "root": str(root),
        "tokenizer_class": tokenizer_class,
        "vocab_size": size,
        "files": files,
    }
    identity["bundle_sha256"] = sha256_json(
        {key: value for key, value in identity.items() if key != "root"}
    )
    return identity


def require_tokenizer_identity(identity: Mapping[str, Any]) -> None:
    """Re-hash tokenizer assets and reject stale or forged attestations."""

    if identity.get("schema_version") != TOKENIZER_ATTESTATION_SCHEMA:
        raise RuntimeError("invalid tokenizer attestation schema")
    root_value = identity.get("root")
    tokenizer_class = identity.get("tokenizer_class")
    if not isinstance(root_value, str) or not root_value:
        raise RuntimeError("invalid tokenizer attestation root")
    if not isinstance(tokenizer_class, str) or not tokenizer_class.strip():
        raise RuntimeError("invalid tokenizer class attestation")
    vocab_size = identity.get("vocab_size")
    try:
        _require_positive_integer(vocab_size, "tokenizer vocab_size")
    except ValueError as error:
        raise RuntimeError(str(error)) from error
    files = identity.get("files")
    if not isinstance(files, list) or not files:
        raise RuntimeError("tokenizer attestation contains no files")
    try:
        root = Path(root_value).resolve(strict=True)
    except FileNotFoundError as error:
        raise RuntimeError("attested tokenizer root no longer exists") from error
    actual_files: list[dict[str, Any]] = []
    seen: set[str] = set()
    attested_relative_paths: list[str] = []
    for record in files:
        if not isinstance(record, dict):
            raise RuntimeError("invalid tokenizer file attestation")
        relative = record.get("relative_path")
        if (
            not isinstance(relative, str)
            or not relative
            or relative in seen
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
        ):
            raise RuntimeError("invalid tokenizer relative path attestation")
        seen.add(relative)
        attested_relative_paths.append(relative)
    discovered_relative_paths = _tokenizer_relative_paths(root)
    if discovered_relative_paths != attested_relative_paths:
        raise RuntimeError(
            "tokenizer asset set changed after attestation: "
            f"{discovered_relative_paths!r} != {attested_relative_paths!r}"
        )
    for record in files:
        relative = record["relative_path"]
        path = _resolve_file(root / relative, "tokenizer asset")
        try:
            path.relative_to(root)
        except ValueError as error:
            raise RuntimeError("tokenizer asset escapes its attested root") from error
        actual_files.append(_stable_tokenizer_file_record(root, path))
    if _tokenizer_relative_paths(root) != attested_relative_paths:
        raise RuntimeError("tokenizer asset set changed while re-hashing")
    if actual_files != files:
        raise RuntimeError("tokenizer asset bytes changed after attestation")
    expected_bundle = sha256_json(
        {
            "schema_version": TOKENIZER_ATTESTATION_SCHEMA,
            "tokenizer_class": tokenizer_class,
            "vocab_size": vocab_size,
            "files": files,
        }
    )
    if identity.get("bundle_sha256") != expected_bundle:
        raise RuntimeError("tokenizer bundle hash does not match its attestation")


def _normalize_token_rows(
    token_rows: Sequence[Sequence[int]], *, vocab_size: int
) -> tuple[list[list[int]], bytes]:
    if isinstance(token_rows, (str, bytes)) or not token_rows:
        raise ValueError("token_rows must contain at least one prompt")
    normalized: list[list[int]] = []
    for row_index, row in enumerate(token_rows):
        if isinstance(row, (str, bytes)) or not row:
            raise ValueError(f"token row {row_index} is empty or invalid")
        normalized_row: list[int] = []
        for token in row:
            if (
                isinstance(token, bool)
                or not isinstance(token, int)
                or token < 0
                or token >= vocab_size
                or token > _INT32_MAX
            ):
                raise ValueError(
                    f"invalid token id in row {row_index}: {token!r}"
                )
            normalized_row.append(token)
        normalized.append(normalized_row)
    serialized = "".join(
        " ".join(str(token) for token in row) + "\n" for row in normalized
    ).encode("ascii")
    return normalized, serialized


def _runtime_library_entries(
    directories: Sequence[str | Path],
) -> list[tuple[str, Path]]:
    if not directories:
        raise ValueError("at least one llama.cpp runtime library directory is required")
    by_name: dict[str, Path] = {}
    by_name_hash: dict[str, str] = {}
    for directory in directories:
        try:
            root = Path(directory).expanduser().resolve(strict=True)
        except FileNotFoundError as error:
            raise ValueError(f"missing runtime library directory: {directory}") from error
        if not root.is_dir():
            raise ValueError(f"runtime library path is not a directory: {root}")
        for entry in sorted(root.iterdir(), key=lambda path: path.name):
            name = entry.name
            if not (
                name.lower().endswith(_RUNTIME_LIBRARY_SUFFIXES)
                or ".so." in name.lower()
                or ".dylib." in name.lower()
            ):
                continue
            resolved = _resolve_file(entry, "runtime library")
            digest = sha256_file(resolved)
            previous = by_name_hash.get(name)
            if previous is not None and previous != digest:
                raise RuntimeError(
                    f"conflicting runtime libraries share the filename {name!r}"
                )
            by_name[name] = resolved
            by_name_hash[name] = digest
    if not by_name:
        raise RuntimeError("no dynamic runtime libraries were discovered")
    return sorted(by_name.items())


def _model_artifact_entries(main_model: Path) -> list[tuple[str, Path]]:
    """Discover one ordinary GGUF or a complete llama.cpp split bundle."""

    match = _GGUF_SPLIT_NAME.fullmatch(main_model.name)
    if match is None:
        return [(main_model.name, main_model)]
    part = int(match.group("part"))
    total = int(match.group("total"))
    if part == 1 and total == 1:
        return [(main_model.name, main_model)]
    if part != 1 or total < 2:
        raise ValueError(
            "split GGUF model_path must name part 00001 of a multi-part bundle"
        )
    prefix = match.group("prefix")
    entries: list[tuple[str, Path]] = []
    for index in range(1, total + 1):
        name = f"{prefix}-{index:05d}-of-{total:05d}.gguf"
        entries.append((name, _resolve_file(main_model.parent / name, "split GGUF part")))
    return entries


def _stat_signature(path: Path) -> tuple[int, int, int, int, int]:
    value = path.stat()
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _file_records(entries: Sequence[tuple[str, Path]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for name, path in entries:
        before = _stat_signature(path)
        digest = sha256_file(path)
        after = _stat_signature(path)
        if before != after:
            raise RuntimeError(f"file changed while hashing: {path}")
        records.append(
            {
                "filename": name,
                "size_bytes": after[2],
                "sha256": digest,
            }
        )
    return records


def _artifact_identity(entries: Sequence[tuple[str, Path]]) -> dict[str, Any]:
    files = _file_records(entries)
    if len(files) == 1:
        # Preserve the established artifact identity for ordinary GGUF files.
        artifact_sha256 = files[0]["sha256"]
    else:
        artifact_sha256 = sha256_json(
            {"schema_version": MODEL_ARTIFACT_SCHEMA, "files": files}
        )
    return {
        "schema_version": MODEL_ARTIFACT_SCHEMA,
        "files": files,
        "artifact_sha256": artifact_sha256,
        "artifact_size_bytes": sum(record["size_bytes"] for record in files),
    }


def _bundle_identity(
    executable: Path, runtime_entries: Sequence[tuple[str, Path]]
) -> dict[str, Any]:
    executable_record = _file_records([(executable.name, executable)])[0]
    libraries = _file_records(runtime_entries)
    return {
        "executable": executable_record,
        "runtime_libraries": libraries,
        "bundle_sha256": sha256_json(
            {"executable": executable_record, "runtime_libraries": libraries}
        ),
    }


def _same_bundle(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return all(
        left.get(key) == right.get(key)
        for key in ("executable", "runtime_libraries", "bundle_sha256")
    )


def _inode_identity_from_stat(value: os.stat_result) -> tuple[int, int]:
    return value.st_dev, value.st_ino


def _unlink_owned_path(path: Path, identity: tuple[int, int]) -> bool:
    try:
        current = os.stat(path, follow_symlinks=False)
    except FileNotFoundError:
        return False
    if _inode_identity_from_stat(current) != identity:
        return False
    try:
        path.unlink()
    except FileNotFoundError:
        return False
    return True


def _write_exclusive(
    path: Path, payload: bytes, *, executable: bool = False
) -> tuple[int, int]:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o700 if executable else 0o600)
    identity = _inode_identity_from_stat(os.fstat(descriptor))
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            descriptor = -1
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        _unlink_owned_path(path, identity)
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return identity


def _link_file(source: Path, destination: Path, *, allow_copy: bool) -> None:
    try:
        os.link(source, destination, follow_symlinks=False)
    except OSError as error:
        if error.errno != errno.EXDEV or not allow_copy:
            raise
        with source.open("rb") as read_stream, destination.open("xb") as write_stream:
            shutil.copyfileobj(read_stream, write_stream, length=1024 * 1024)
            write_stream.flush()
            os.fsync(write_stream.fileno())
        shutil.copymode(source, destination)


def _path_exists(path: Path) -> bool:
    return os.path.lexists(path)


def _sync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _load_summary(stdout: str) -> dict[str, Any]:
    lines = [line for line in stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError(
            "native raw-logit collector must emit exactly one summary object"
        )
    try:
        value = json.loads(lines[0])
    except json.JSONDecodeError as error:
        raise RuntimeError("invalid native raw-logit collector summary") from error
    if not isinstance(value, dict):
        raise RuntimeError("native raw-logit collector summary is not an object")
    return value


def _validate_summary(
    summary: Mapping[str, Any],
    *,
    count: int,
    vocab_size: int,
    maximum_prompt_tokens: int,
    threads: int,
    gpu_layers: int,
) -> None:
    exact = {
        "schema_version": NATIVE_COLLECTOR_SCHEMA,
        "count": count,
        "vocab_size": vocab_size,
        "values_per_row": vocab_size,
        "maximum_prompt_tokens": maximum_prompt_tokens,
        "threads": threads,
        "gpu_layers": gpu_layers,
        "float_format": RAW_LOGIT_FORMAT,
        "memory_cleared_between_rows": True,
        "backend_loading": "explicit-directory",
    }
    for key, expected in exact.items():
        if summary.get(key) != expected:
            raise RuntimeError(
                f"invalid native collector summary {key}: "
                f"{summary.get(key)!r} != {expected!r}"
            )
    gpu_offload_supported = summary.get("gpu_offload_supported")
    if not isinstance(gpu_offload_supported, bool):
        raise RuntimeError("invalid native collector GPU-offload capability")
    if gpu_layers != 0 and not gpu_offload_supported:
        raise RuntimeError(
            "native collector silently lacks requested GPU layer offload"
        )
    for key in ("n_ctx", "n_batch", "n_ubatch", "model_context_train"):
        value = summary.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise RuntimeError(f"invalid native collector summary {key}")
    if summary["n_ctx"] < maximum_prompt_tokens:
        raise RuntimeError("native collector context truncated the prompt set")
    if summary["n_batch"] < maximum_prompt_tokens:
        raise RuntimeError("native collector batch truncated the prompt set")
    if summary["n_ubatch"] > summary["n_batch"]:
        raise RuntimeError("native collector ubatch exceeds its logical batch")
    if summary["model_context_train"] < maximum_prompt_tokens:
        raise RuntimeError("prompt set exceeds the model training context")
    for key in ("model_size_bytes", "model_parameter_count"):
        value = summary.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise RuntimeError(f"invalid native collector summary {key}")
    for key in ("model_description", "system_info"):
        value = summary.get(key)
        if not isinstance(value, str) or not value.strip():
            raise RuntimeError(f"invalid native collector summary {key}")
    expected_inventory_method = (
        "windows-toolhelp32"
        if os.name == "nt"
        else "macos-dyld-images"
        if sys.platform == "darwin"
        else "linux-proc-self-maps"
        if sys.platform.startswith("linux")
        else None
    )
    if expected_inventory_method is None:
        raise RuntimeError(
            f"native runtime module attestation is unsupported on {sys.platform}"
        )
    if summary.get("module_inventory_method") != expected_inventory_method:
        raise RuntimeError("invalid native collector module inventory method")
    loaded_modules = summary.get("loaded_modules")
    if (
        not isinstance(loaded_modules, list)
        or not loaded_modules
        or any(not isinstance(path, str) or not path for path in loaded_modules)
        or len(set(loaded_modules)) != len(loaded_modules)
    ):
        raise RuntimeError("invalid native collector loaded-module inventory")
    for key in ("model_load_seconds", "decode_seconds", "total_seconds"):
        value = summary.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise RuntimeError(f"invalid native collector summary {key}")


def _runtime_module_family(filename: str) -> str | None:
    name = filename.lower()
    if name.startswith("libllama") or name in {"llama.dll", "libllama.dll"}:
        return "llama"
    if name.startswith("libggml") or name.startswith("ggml-") or name in {
        "ggml.dll",
        "libggml.dll",
    }:
        return "ggml"
    return None


def _gpu_backend_family(filename: str) -> str | None:
    name = filename.lower()
    for family in ("metal", "cuda", "hip", "musa", "sycl", "vulkan", "kompute"):
        if family in name and _runtime_module_family(name) == "ggml":
            return family
    return None


def _attest_loaded_modules(
    summary: Mapping[str, Any],
    *,
    library_root: Path,
    executable: Path,
    bundle: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Bind the loader's actual module inventory to pinned runtime bytes."""

    expected_libraries = {
        record["filename"]: record for record in bundle["runtime_libraries"]
    }
    executable_record = bundle["executable"]
    normalized: list[dict[str, Any]] = []
    loaded_families: set[str] = set()
    loaded_gpu_backends: set[str] = set()
    root = library_root.resolve(strict=True)
    collector = executable.resolve(strict=True)
    for raw_path in summary["loaded_modules"]:
        if raw_path.endswith(" (deleted)"):
            raise RuntimeError("native collector loaded a deleted runtime module")
        path = Path(raw_path)
        if not path.is_absolute():
            raise RuntimeError("native collector reported a non-absolute module path")
        try:
            resolved = path.resolve(strict=True)
        except FileNotFoundError as error:
            if _runtime_module_family(path.name) is not None:
                raise RuntimeError(
                    f"loaded llama.cpp runtime module is no longer resolvable: {path}"
                ) from error
            normalized.append({"scope": "platform", "path": raw_path})
            continue
        if resolved.samefile(collector):
            normalized.append(
                {"scope": "pinned-collector", **dict(executable_record)}
            )
            continue
        try:
            resolved.relative_to(root)
        except ValueError:
            family = _runtime_module_family(resolved.name)
            if family is not None:
                raise RuntimeError(
                    "llama.cpp runtime module escaped the pinned runtime directory: "
                    f"{resolved}"
                )
            normalized.append({"scope": "platform", "path": str(resolved)})
            continue
        expected = expected_libraries.get(resolved.name)
        if expected is None:
            raise RuntimeError(
                f"loaded runtime module was not attested: {resolved.name}"
            )
        actual = _file_records([(resolved.name, resolved)])[0]
        if actual != expected:
            raise RuntimeError(
                f"loaded runtime module bytes differ from attestation: {resolved.name}"
            )
        family = _runtime_module_family(resolved.name)
        if family is not None:
            loaded_families.add(family)
        gpu_backend = _gpu_backend_family(resolved.name)
        if gpu_backend is not None:
            loaded_gpu_backends.add(gpu_backend)
        normalized.append({"scope": "pinned-runtime", **actual})
    if loaded_families != {"llama", "ggml"}:
        raise RuntimeError(
            "loaded module inventory does not contain pinned llama and ggml runtimes"
        )
    if summary["gpu_layers"] != 0 and not loaded_gpu_backends:
        raise RuntimeError(
            "GPU offload was requested but no pinned GPU backend module was loaded"
        )
    return sorted(normalized, key=canonical_json)


def _validate_recorded_runtime_protocol(
    progress: Mapping[str, Any], bundle: Mapping[str, Any]
) -> None:
    protocol = progress.get("runtime_protocol")
    if not isinstance(protocol, dict):
        raise RuntimeError("stale native raw-logit runtime protocol")
    protocol_sha256 = sha256_json(protocol)
    if (
        progress.get("runtime_protocol_sha256") != protocol_sha256
        or protocol.get("collector_bundle_sha256") != bundle.get("bundle_sha256")
        or progress.get("runtime_model", {}).get("build_info")
        != f"native-{protocol_sha256}"
    ):
        raise RuntimeError("stale native raw-logit runtime protocol hash")
    modules = protocol.get("loaded_modules")
    if not isinstance(modules, list) or not modules:
        raise RuntimeError("stale native raw-logit loaded-module attestation")
    expected_libraries = {
        record["filename"]: record for record in bundle["runtime_libraries"]
    }
    families: set[str] = set()
    gpu_backends: set[str] = set()
    collector_seen = False
    for module in modules:
        if not isinstance(module, dict):
            raise RuntimeError("invalid recorded native runtime module")
        scope = module.get("scope")
        if scope == "pinned-collector":
            if collector_seen or {
                key: value for key, value in module.items() if key != "scope"
            } != bundle["executable"]:
                raise RuntimeError("invalid recorded native collector module")
            collector_seen = True
        elif scope == "pinned-runtime":
            record = {key: value for key, value in module.items() if key != "scope"}
            if expected_libraries.get(record.get("filename")) != record:
                raise RuntimeError("invalid recorded native runtime library")
            family = _runtime_module_family(str(record["filename"]))
            if family is not None:
                families.add(family)
            gpu_backend = _gpu_backend_family(str(record["filename"]))
            if gpu_backend is not None:
                gpu_backends.add(gpu_backend)
        elif scope == "platform":
            if not isinstance(module.get("path"), str) or not module["path"]:
                raise RuntimeError("invalid recorded platform runtime module")
        else:
            raise RuntimeError("invalid recorded native runtime module scope")
    if not collector_seen or families != {"llama", "ggml"}:
        raise RuntimeError("incomplete recorded native runtime closure")
    gpu_layers = protocol.get("gpu_layers")
    gpu_supported = protocol.get("gpu_offload_supported")
    if not isinstance(gpu_layers, int) or isinstance(gpu_layers, bool):
        raise RuntimeError("invalid recorded native GPU-layer request")
    if not isinstance(gpu_supported, bool):
        raise RuntimeError("invalid recorded native GPU-offload capability")
    if gpu_layers != 0 and (not gpu_supported or not gpu_backends):
        raise RuntimeError("incomplete recorded native GPU runtime closure")


def _requested_protocol(
    *,
    context_size: int | None,
    batch_size: int | None,
    ubatch_size: int | None,
    threads: int,
    gpu_layers: int,
) -> dict[str, Any]:
    return {
        "context_size": context_size,
        "batch_size": batch_size,
        "ubatch_size": ubatch_size,
        "threads": threads,
        "gpu_layers": gpu_layers,
    }


def _validate_existing(
    *,
    data_path: Path,
    progress_path: Path,
    schema_version: str,
    label: str,
    model_alias: str,
    model_path: Path,
    artifact_identity: Mapping[str, Any],
    tokenizer_identity: Mapping[str, Any],
    token_rows_sha256: str,
    token_file_sha256: str,
    count: int,
    vocab_size: int,
    maximum_prompt_tokens: int,
    bundle: Mapping[str, Any],
    requested_protocol: Mapping[str, Any],
) -> NativeRawLogitsResult:
    matrix, progress = load_completed_raw_logits(
        data_path,
        progress_path,
        schema_version=schema_version,
        count=count,
        vocab_size=vocab_size,
    )
    del matrix
    expected = {
        "label": label,
        "model": model_alias,
        "prompt_tokens_sha256": token_rows_sha256,
        "token_file_sha256": token_file_sha256,
        "tokenizer": dict(tokenizer_identity),
        "collector_request": dict(requested_protocol),
    }
    for key, value in expected.items():
        if progress.get(key) != value:
            raise RuntimeError(f"stale native raw-logit artifact: mismatched {key}")
    collector = progress.get("collector")
    if not isinstance(collector, dict) or not _same_bundle(collector, bundle):
        raise RuntimeError("stale native raw-logit runtime bundle")
    summary = progress.get("collector_summary")
    if not isinstance(summary, dict):
        raise RuntimeError("stale native raw-logit collector summary")
    _validate_summary(
        summary,
        count=count,
        vocab_size=vocab_size,
        maximum_prompt_tokens=maximum_prompt_tokens,
        threads=int(requested_protocol["threads"]),
        gpu_layers=int(requested_protocol["gpu_layers"]),
    )
    _validate_recorded_runtime_protocol(progress, bundle)
    if (
        progress.get("artifact_sha256")
        != artifact_identity.get("artifact_sha256")
        or progress.get("runtime_model", {}).get("artifact_size_bytes")
        != artifact_identity.get("artifact_size_bytes")
        or progress.get("runtime_model", {}).get("artifact_files")
        != artifact_identity.get("files")
    ):
        raise RuntimeError("stale native raw-logit model artifact")
    attested_path = progress.get("runtime_model", {}).get("model_path")
    try:
        if not Path(str(attested_path)).resolve(strict=True).samefile(model_path):
            raise RuntimeError("stale native raw-logit model path")
    except FileNotFoundError as error:
        raise RuntimeError("attested native raw-logit model path is missing") from error
    return NativeRawLogitsResult(data_path, progress_path, progress, True)


def collect_native_raw_logits(
    *,
    token_rows: Sequence[Sequence[int]],
    tokenizer_identity: Mapping[str, Any],
    model_path: str | Path,
    output_path: str | Path,
    schema_version: str,
    label: str,
    model_alias: str,
    executable_path: str | Path,
    runtime_library_dirs: Sequence[str | Path],
    progress_path: str | Path | None = None,
    context_size: int | None = None,
    batch_size: int | None = None,
    ubatch_size: int | None = None,
    threads: int = 4,
    gpu_layers: int = -1,
    timeout_seconds: float | None = None,
) -> NativeRawLogitsResult:
    """Collect full-vocabulary first-token logits through llama.cpp's C API.

    Existing matching artifacts are immutable cache hits.  Any partial, stale,
    or conflicting destination fails closed; this function never truncates or
    overwrites an output selected by the caller.
    """

    started = time.perf_counter()
    if not isinstance(schema_version, str) or not schema_version.strip():
        raise ValueError("schema_version must be a non-empty string")
    if not isinstance(label, str) or not label.strip():
        raise ValueError("label must be a non-empty string")
    if not isinstance(model_alias, str) or not model_alias.strip():
        raise ValueError("model_alias must be a non-empty string")
    threads = _require_positive_integer(threads, "threads")
    if threads > _INT32_MAX:
        raise ValueError("threads exceeds llama.cpp's int32 range")
    if (
        isinstance(gpu_layers, bool)
        or not isinstance(gpu_layers, int)
        or not _INT32_MIN <= gpu_layers <= _INT32_MAX
    ):
        raise ValueError("gpu_layers must fit llama.cpp's int32 range")
    for name, value in (
        ("context_size", context_size),
        ("batch_size", batch_size),
        ("ubatch_size", ubatch_size),
    ):
        if value is not None:
            _require_positive_integer(value, name)
            if value > _UINT32_MAX:
                raise ValueError(f"{name} exceeds llama.cpp's uint32 range")
    if timeout_seconds is not None and (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(float(timeout_seconds))
        or timeout_seconds <= 0
    ):
        raise ValueError("timeout_seconds must be finite and positive")

    require_tokenizer_identity(tokenizer_identity)
    vocab_size = _require_positive_integer(
        tokenizer_identity.get("vocab_size"), "tokenizer vocab_size"
    )
    if vocab_size > _INT32_MAX:
        raise ValueError("tokenizer vocab_size exceeds llama.cpp's int32 range")
    normalized_rows, serialized_tokens = _normalize_token_rows(
        token_rows, vocab_size=vocab_size
    )
    count = len(normalized_rows)
    if count > _INT32_MAX:
        raise ValueError("token row count exceeds llama.cpp's int32 range")
    maximum_prompt_tokens = max(map(len, normalized_rows))
    if context_size is not None and context_size < maximum_prompt_tokens:
        raise ValueError("context_size is smaller than the longest prompt")
    if batch_size is not None and batch_size < maximum_prompt_tokens:
        raise ValueError("batch_size is smaller than the longest prompt")
    if (
        batch_size is not None
        and ubatch_size is not None
        and ubatch_size > batch_size
    ):
        raise ValueError("ubatch_size cannot exceed batch_size")

    model = _resolve_file(model_path, "GGUF model")
    model_entries = _model_artifact_entries(model)
    source_artifact = _artifact_identity(model_entries)
    executable = _resolve_file(executable_path, "native collector executable")
    if not os.access(executable, os.X_OK):
        raise ValueError(f"native collector is not executable: {executable}")
    runtime_entries = _runtime_library_entries(runtime_library_dirs)
    source_bundle = _bundle_identity(executable, runtime_entries)
    require_tokenizer_identity(tokenizer_identity)

    output = Path(output_path).expanduser().resolve(strict=False)
    progress = (
        Path(progress_path).expanduser().resolve(strict=False)
        if progress_path is not None
        else default_progress_path(output)
    )
    if output == progress:
        raise ValueError("raw-logit data and progress paths must be distinct")
    output.parent.mkdir(parents=True, exist_ok=True)
    progress.parent.mkdir(parents=True, exist_ok=True)
    token_rows_sha256 = sha256_json(normalized_rows)
    token_file_sha256 = sha256_bytes(serialized_tokens)
    requested_protocol = _requested_protocol(
        context_size=context_size,
        batch_size=batch_size,
        ubatch_size=ubatch_size,
        threads=threads,
        gpu_layers=gpu_layers,
    )

    output_exists = _path_exists(output)
    progress_exists = _path_exists(progress)
    if output_exists or progress_exists:
        if not output_exists or not progress_exists:
            raise RuntimeError(
                "partial native raw-logit destination exists; refusing to overwrite it"
            )
        return _validate_existing(
            data_path=output,
            progress_path=progress,
            schema_version=schema_version,
            label=label,
            model_alias=model_alias,
            model_path=model,
            artifact_identity=source_artifact,
            tokenizer_identity=tokenizer_identity,
            token_rows_sha256=token_rows_sha256,
            token_file_sha256=token_file_sha256,
            count=count,
            vocab_size=vocab_size,
            maximum_prompt_tokens=maximum_prompt_tokens,
            bundle=source_bundle,
            requested_protocol=requested_protocol,
        )

    lock_path = progress.with_suffix(progress.suffix + ".lock")
    try:
        lock_identity = _write_exclusive(
            lock_path,
            canonical_json(
                {
                    "pid": os.getpid(),
                    "output": str(output),
                    "progress": str(progress),
                }
            )
            + b"\n",
        )
    except FileExistsError as error:
        raise RuntimeError(f"native raw-logit destination is locked: {lock_path}") from error

    published_output_identity: tuple[int, int] | None = None
    published_progress_identity: tuple[int, int] | None = None
    try:
        with tempfile.TemporaryDirectory(
            prefix=".native-logits-", dir=output.parent
        ) as temporary_value:
            temporary = Path(temporary_value)
            source_model_stats: dict[str, tuple[int, int, int, int, int]] = {}
            linked_model_entries: list[tuple[str, Path]] = []
            for name, source in model_entries:
                destination = temporary / name
                try:
                    _link_file(source, destination, allow_copy=False)
                except OSError as error:
                    if error.errno == errno.EXDEV:
                        raise RuntimeError(
                            "model and raw-logit destination must share a filesystem "
                            "so the evaluated inode can be pinned without copying"
                        ) from error
                    raise
                source_after_link = _stat_signature(source)
                if _stat_signature(destination) != source_after_link:
                    raise RuntimeError("pinned GGUF inode differs from its source")
                source_model_stats[name] = source_after_link
                linked_model_entries.append((name, destination))
            linked_model = linked_model_entries[0][1]
            linked_executable = temporary / executable.name
            _link_file(executable, linked_executable, allow_copy=True)
            # ggml's dynamic backend loader searches beside the executable.
            # Keep the pinned aliases there as well as in the loader path.
            library_root = temporary
            linked_runtime_entries: list[tuple[str, Path]] = []
            for name, source in runtime_entries:
                destination = library_root / name
                _link_file(source, destination, allow_copy=True)
                linked_runtime_entries.append((name, destination))
            linked_bundle_before = _bundle_identity(
                linked_executable, linked_runtime_entries
            )
            if not _same_bundle(source_bundle, linked_bundle_before):
                raise RuntimeError("pinned runtime bundle differs from its source")

            tokens_path = temporary / "tokens.txt"
            _write_exclusive(tokens_path, serialized_tokens)
            pinned_token_sha256 = sha256_file(tokens_path)
            artifact_sha256 = source_artifact["artifact_sha256"]
            artifact_size_bytes = source_artifact["artifact_size_bytes"]
            raw_path = temporary / "logits.partial.bin"
            manifest_path = temporary / "logits.progress.json"
            command = [
                str(linked_executable),
                str(linked_model),
                str(tokens_path),
                str(raw_path),
                "--threads",
                str(threads),
                "--gpu-layers",
                str(gpu_layers),
                "--expected-count",
                str(count),
                "--expected-vocab",
                str(vocab_size),
                "--backend-dir",
                str(library_root),
            ]
            if context_size is not None:
                command.extend(("--n-ctx", str(context_size)))
            if batch_size is not None:
                command.extend(("--n-batch", str(batch_size)))
            if ubatch_size is not None:
                command.extend(("--n-ubatch", str(ubatch_size)))

            environment = os.environ.copy()
            library_value = str(library_root)
            environment["DYLD_LIBRARY_PATH"] = library_value
            environment["LD_LIBRARY_PATH"] = library_value
            environment["PATH"] = library_value + os.pathsep + environment.get(
                "PATH", ""
            )
            process_started = time.perf_counter()
            try:
                completed = subprocess.run(
                    command,
                    check=False,
                    capture_output=True,
                    text=True,
                    env=environment,
                    timeout=timeout_seconds,
                )
            except subprocess.TimeoutExpired as error:
                raise RuntimeError("native raw-logit collection timed out") from error
            process_seconds = time.perf_counter() - process_started
            if completed.returncode != 0:
                stderr_tail = "\n".join(completed.stderr.splitlines()[-20:])
                raise RuntimeError(
                    "native raw-logit collector failed with exit code "
                    f"{completed.returncode}:\n{stderr_tail}"
                )
            summary = _load_summary(completed.stdout)
            _validate_summary(
                summary,
                count=count,
                vocab_size=vocab_size,
                maximum_prompt_tokens=maximum_prompt_tokens,
                threads=threads,
                gpu_layers=gpu_layers,
            )
            expected_bytes = count * vocab_size * 4
            try:
                raw_bytes = raw_path.stat().st_size
            except FileNotFoundError as error:
                raise RuntimeError("native collector did not produce raw logits") from error
            if raw_bytes != expected_bytes:
                raise RuntimeError(
                    f"native raw-logit size mismatch: {raw_bytes} != {expected_bytes}"
                )

            linked_artifact_after = _artifact_identity(linked_model_entries)
            if linked_artifact_after != source_artifact:
                raise RuntimeError("GGUF model bytes changed during native collection")
            for name, linked_path in linked_model_entries:
                if _stat_signature(linked_path) != source_model_stats[name]:
                    raise RuntimeError("GGUF model inode changed during native collection")
            linked_bundle_after = _bundle_identity(
                linked_executable, linked_runtime_entries
            )
            if not _same_bundle(linked_bundle_before, linked_bundle_after):
                raise RuntimeError("native runtime bytes changed during collection")
            loaded_module_attestation = _attest_loaded_modules(
                summary,
                library_root=library_root,
                executable=linked_executable,
                bundle=linked_bundle_after,
            )
            if sha256_file(tokens_path) != pinned_token_sha256:
                raise RuntimeError("token input bytes changed during native collection")
            require_tokenizer_identity(tokenizer_identity)

            data_sha256 = sha256_file(raw_path)
            runtime_protocol = {
                "schema_version": RUNTIME_PROTOCOL_SCHEMA,
                "collector_bundle_sha256": linked_bundle_after["bundle_sha256"],
                "collector_schema_version": summary["schema_version"],
                "system_info": summary["system_info"],
                "module_inventory_method": summary["module_inventory_method"],
                "loaded_modules": loaded_module_attestation,
                "float_format": summary["float_format"],
                "memory_cleared_between_rows": summary[
                    "memory_cleared_between_rows"
                ],
                "backend_loading": summary["backend_loading"],
                "n_ctx": summary["n_ctx"],
                "n_batch": summary["n_batch"],
                "n_ubatch": summary["n_ubatch"],
                "threads": summary["threads"],
                "gpu_layers": summary["gpu_layers"],
                "gpu_offload_supported": summary["gpu_offload_supported"],
            }
            runtime_protocol_sha256 = sha256_json(runtime_protocol)
            runtime_model = {
                "endpoint": "native://llama-raw-logits",
                "model_alias": model_alias,
                "model_ftype": summary["model_description"],
                "model_path": str(model),
                "artifact_sha256": artifact_sha256,
                "artifact_size_bytes": artifact_size_bytes,
                "artifact_files": source_artifact["files"],
                "build_info": f"native-{runtime_protocol_sha256}",
            }
            manifest = {
                "schema_version": schema_version,
                "label": label,
                "model": model_alias,
                "prompt_tokens_sha256": token_rows_sha256,
                "token_file_sha256": token_file_sha256,
                "tokenizer": dict(tokenizer_identity),
                "vocab_size": vocab_size,
                "count": count,
                "completed": count,
                "artifact_sha256": artifact_sha256,
                "artifact_attested": True,
                "artifact_files": source_artifact["files"],
                "runtime_model": runtime_model,
                "collector": linked_bundle_after,
                "collector_request": requested_protocol,
                "runtime_protocol": runtime_protocol,
                "runtime_protocol_sha256": runtime_protocol_sha256,
                "collector_summary": summary,
                "data_format": RAW_LOGIT_FORMAT,
                "data_size_bytes": raw_bytes,
                "data_sha256": data_sha256,
                "maximum_prompt_tokens": maximum_prompt_tokens,
                "process_seconds": process_seconds,
                "seconds": time.perf_counter() - started,
            }
            _write_exclusive(manifest_path, canonical_json(manifest) + b"\n")
            validated, _ = load_completed_raw_logits(
                raw_path,
                manifest_path,
                schema_version=schema_version,
                count=count,
                vocab_size=vocab_size,
            )
            del validated

            try:
                raw_identity = _inode_identity_from_stat(
                    os.stat(raw_path, follow_symlinks=False)
                )
                os.link(raw_path, output)
                published_output_identity = raw_identity
                manifest_identity = _inode_identity_from_stat(
                    os.stat(manifest_path, follow_symlinks=False)
                )
                os.link(manifest_path, progress)
                published_progress_identity = manifest_identity
                _sync_directory(output.parent)
                if progress.parent != output.parent:
                    _sync_directory(progress.parent)
            except BaseException:
                if published_progress_identity is not None:
                    _unlink_owned_path(progress, published_progress_identity)
                    published_progress_identity = None
                if published_output_identity is not None:
                    _unlink_owned_path(output, published_output_identity)
                    published_output_identity = None
                raise

        matrix, loaded = load_completed_raw_logits(
            output,
            progress,
            schema_version=schema_version,
            count=count,
            vocab_size=vocab_size,
        )
        del matrix
        return NativeRawLogitsResult(output, progress, loaded, False)
    finally:
        _unlink_owned_path(lock_path, lock_identity)
