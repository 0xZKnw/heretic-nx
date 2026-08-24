"""Deterministic content hashing primitives."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
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


def sha256_directory(path: str | Path) -> str:
    """Hash a directory from sorted relative paths and individual file hashes."""

    root = Path(path)
    if not root.is_dir():
        raise ValueError(f"not a directory: {root}")
    entries = [
        {"path": file.relative_to(root).as_posix(), "sha256": sha256_file(file)}
        for file in sorted((item for item in root.rglob("*") if item.is_file()), key=lambda item: item.relative_to(root).as_posix())
    ]
    if not entries:
        raise ValueError("cannot hash an empty directory")
    return sha256_json(entries)
