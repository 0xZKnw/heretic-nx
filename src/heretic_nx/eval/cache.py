"""Content-addressed SQLite cache for judge decisions."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from heretic_nx.hashing import sha256_json


def judge_cache_key(prompt: str, response: str, rubric: str) -> str:
    return sha256_json({"prompt": prompt, "response": response, "rubric": rubric})


class JudgeCache:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS verdicts (cache_key TEXT PRIMARY KEY, payload TEXT NOT NULL)"
        )
        self.connection.commit()

    def get(self, cache_key: str) -> dict[str, object] | None:
        row = self.connection.execute(
            "SELECT payload FROM verdicts WHERE cache_key = ?", (cache_key,)
        ).fetchone()
        return None if row is None else json.loads(row[0])

    def put(self, cache_key: str, payload: dict[str, object]) -> None:
        self.connection.execute(
            "INSERT OR REPLACE INTO verdicts(cache_key, payload) VALUES(?, ?)",
            (cache_key, json.dumps(payload, sort_keys=True, ensure_ascii=False)),
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "JudgeCache":
        return self

    def __exit__(self, *_args) -> None:
        self.close()
