"""Deterministic content hashing primitives."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any


_DEFAULT_DIRECTORY_HASH_WORKERS = 2


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value))


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_directory(path: str | Path, *, workers: int | None = None) -> str:
    """Hash a directory from sorted relative paths and individual file hashes.

    Independent files are read concurrently by default.  ``executor.map`` keeps
    their digests in canonical path order, so concurrency cannot affect the
    directory digest.
    """

    root = Path(path)
    if not root.is_dir():
        raise ValueError(f"not a directory: {root}")
    if workers is not None and (isinstance(workers, bool) or not isinstance(workers, int) or workers < 1):
        raise ValueError("workers must be a positive integer or None")
    files = sorted(
        (item for item in root.rglob("*") if item.is_file()),
        key=lambda item: item.relative_to(root).as_posix(),
    )
    if not files:
        raise ValueError("cannot hash an empty directory")
    worker_count = min(workers or _DEFAULT_DIRECTORY_HASH_WORKERS, len(files))
    if worker_count == 1:
        digests = [sha256_file(file) for file in files]
    else:
        with ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="hnx-sha256",
        ) as executor:
            digests = list(executor.map(sha256_file, files))
    entries = [
        {"path": file.relative_to(root).as_posix(), "sha256": digest}
        for file, digest in zip(files, digests, strict=True)
    ]
    return sha256_json(entries)
