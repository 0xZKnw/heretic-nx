"""Content-addressed SQLite cache for judge decisions.

Single writes are durable before :meth:`put` returns. Callers evaluating a
collection can opt into :meth:`put_many` or :meth:`transaction` to amortize
SQLite commits without relying on a background write-behind queue.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path

from heretic_nx.hashing import sha256_json


class JudgeCacheError(RuntimeError):
    """Base class for errors that must not be interpreted as cache misses."""


class JudgeCacheConflictError(JudgeCacheError):
    """Raised when a content key is replayed with a different payload."""


class JudgeCacheCorruptionError(JudgeCacheError):
    """Raised when a durable cache row cannot be decoded safely."""


def judge_cache_key(
    prompt: str,
    response: str,
    rubric: str,
    *,
    task_success: bool | None = None,
) -> str:
    # Keep the historical key for the overwhelmingly common ``None`` case so
    # existing caches remain reusable. Task-specific success changes the J0
    # verdict, so it must be bound into new keys when supplied.
    material: dict[str, object] = {
        "prompt": prompt,
        "response": response,
        "rubric": rubric,
    }
    if task_success is not None:
        material["task_success"] = task_success
    return sha256_json(material)


def _encode_payload(payload: Mapping[str, object]) -> str:
    try:
        return json.dumps(
            dict(payload),
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as error:
        raise JudgeCacheError("judge cache payload is not finite JSON") from error


def _reject_nonfinite_json(value: str) -> object:
    raise ValueError(f"non-finite JSON constant: {value}")


class JudgeCache:
    """Durable, immutable mapping from content hashes to judge payloads.

    A key is write-once: replaying the same payload is idempotent, while a
    different payload raises :class:`JudgeCacheConflictError`. This prevents
    nondeterministic or concurrent judges from silently changing prior
    evidence.
    """

    _QUERY_CHUNK_SIZE = 900

    def __init__(self, path: str | Path, *, timeout: float = 30.0) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path, timeout=timeout)
        # WAL lets readers replay committed verdicts while a batch writer is
        # active. FULL retains SQLite's durable-commit guarantee.
        self.connection.execute("PRAGMA journal_mode = WAL")
        self.connection.execute("PRAGMA synchronous = FULL")
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS verdicts "
            "(cache_key TEXT PRIMARY KEY, payload TEXT NOT NULL)"
        )
        self.connection.commit()
        self._managed_transaction = False

    def _decode_payload(self, cache_key: str, encoded: object) -> dict[str, object]:
        if not isinstance(encoded, str):
            raise JudgeCacheCorruptionError(
                f"judge cache payload for {cache_key} is not text"
            )
        try:
            payload = json.loads(
                encoded,
                parse_constant=_reject_nonfinite_json,
            )
        except (ValueError, TypeError) as error:
            raise JudgeCacheCorruptionError(
                f"judge cache payload for {cache_key} is invalid JSON"
            ) from error
        if not isinstance(payload, dict):
            raise JudgeCacheCorruptionError(
                f"judge cache payload for {cache_key} is not an object"
            )
        return payload

    def get(self, cache_key: str) -> dict[str, object] | None:
        row = self.connection.execute(
            "SELECT payload FROM verdicts WHERE cache_key = ?", (cache_key,)
        ).fetchone()
        return None if row is None else self._decode_payload(cache_key, row[0])

    def get_many(self, cache_keys: Iterable[str]) -> dict[str, dict[str, object]]:
        """Fetch unique keys with a bounded number of SQLite statements."""

        keys = tuple(dict.fromkeys(cache_keys))
        found: dict[str, dict[str, object]] = {}
        for offset in range(0, len(keys), self._QUERY_CHUNK_SIZE):
            chunk = keys[offset : offset + self._QUERY_CHUNK_SIZE]
            placeholders = ",".join("?" for _ in chunk)
            rows = self.connection.execute(
                f"SELECT cache_key, payload FROM verdicts "
                f"WHERE cache_key IN ({placeholders})",
                chunk,
            ).fetchall()
            for cache_key, encoded in rows:
                found[cache_key] = self._decode_payload(cache_key, encoded)
        return found

    def _prepare_entries(
        self,
        entries: Iterable[tuple[str, Mapping[str, object]]],
    ) -> tuple[tuple[str, str], ...]:
        prepared: dict[str, str] = {}
        for cache_key, payload in entries:
            encoded = _encode_payload(payload)
            prior = prepared.setdefault(cache_key, encoded)
            if prior != encoded:
                raise JudgeCacheConflictError(
                    f"batch contains conflicting payloads for {cache_key}"
                )
        return tuple(prepared.items())

    def _put_prepared(self, entries: tuple[tuple[str, str], ...]) -> None:
        if not entries:
            return

        keys = tuple(cache_key for cache_key, _payload in entries)
        existing = self.get_many(keys)
        new_rows: list[tuple[str, str]] = []
        for cache_key, encoded in entries:
            durable = existing.get(cache_key)
            if durable is None:
                new_rows.append((cache_key, encoded))
                continue
            if _encode_payload(durable) != encoded:
                raise JudgeCacheConflictError(
                    f"immutable judge cache conflict for {cache_key}"
                )

        # transaction() uses BEGIN IMMEDIATE, so no other writer can insert a
        # key between the preflight read and this insert.
        self.connection.executemany(
            "INSERT INTO verdicts(cache_key, payload) VALUES(?, ?)", new_rows
        )

    @contextmanager
    def transaction(self) -> Iterator["JudgeCache"]:
        """Group explicit writes into one durable, rollback-safe transaction.

        Nested managed transactions are rejected instead of introducing
        ambiguous partial-commit semantics.
        """

        if self._managed_transaction or self.connection.in_transaction:
            raise JudgeCacheError("nested judge cache transactions are not supported")
        self.connection.execute("BEGIN IMMEDIATE")
        self._managed_transaction = True
        try:
            yield self
        except BaseException:
            self.connection.rollback()
            raise
        else:
            self.connection.commit()
        finally:
            self._managed_transaction = False

    def put(self, cache_key: str, payload: Mapping[str, object]) -> None:
        prepared = self._prepare_entries(((cache_key, payload),))
        if self._managed_transaction:
            self._put_prepared(prepared)
            return
        with self.transaction():
            self._put_prepared(prepared)

    def put_many(
        self,
        entries: Iterable[tuple[str, Mapping[str, object]]],
    ) -> None:
        """Atomically persist many immutable entries with one commit."""

        prepared = self._prepare_entries(entries)
        if not prepared:
            return
        if self._managed_transaction:
            self._put_prepared(prepared)
            return
        with self.transaction():
            self._put_prepared(prepared)

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "JudgeCache":
        return self

    def __exit__(self, *_args) -> None:
        self.close()
