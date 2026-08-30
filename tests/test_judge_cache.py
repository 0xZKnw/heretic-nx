from __future__ import annotations

import sqlite3
import threading

import pytest

from heretic_nx.eval.cache import (
    JudgeCache,
    JudgeCacheConflictError,
    JudgeCacheCorruptionError,
    judge_cache_key,
)
from heretic_nx.eval.cascade import JudgeCascade, JudgeInput, JudgeVerdict


def _payload(label: str = "compliance") -> dict[str, object]:
    return {
        "label": label,
        "confidence": 0.95,
        "level": "J1",
        "rationale": "test judge",
    }


def test_cache_is_immutable_and_idempotent_across_reopen(tmp_path) -> None:
    path = tmp_path / "judge.sqlite"
    with JudgeCache(path) as cache:
        cache.put("key", _payload())
        cache.put("key", _payload())
        with pytest.raises(JudgeCacheConflictError):
            cache.put("key", _payload("refusal"))

    with JudgeCache(path) as reopened:
        assert reopened.get("key") == _payload()


def test_put_many_is_atomic_when_a_replay_conflicts(tmp_path) -> None:
    path = tmp_path / "judge.sqlite"
    with JudgeCache(path) as cache:
        cache.put("durable", _payload())
        with pytest.raises(JudgeCacheConflictError):
            cache.put_many(
                (
                    ("new", _payload()),
                    ("durable", _payload("refusal")),
                )
            )
        assert cache.get("new") is None
        assert cache.get("durable") == _payload()


def test_explicit_transaction_rolls_back_and_commits_durably(tmp_path) -> None:
    path = tmp_path / "judge.sqlite"
    with JudgeCache(path) as cache:
        with pytest.raises(RuntimeError, match="abort"):
            with cache.transaction():
                cache.put("rolled-back", _payload())
                raise RuntimeError("abort")
        assert cache.get("rolled-back") is None

        with cache.transaction():
            cache.put("one", _payload())
            cache.put("two", _payload("refusal"))

    with JudgeCache(path) as reopened:
        assert reopened.get_many(("one", "two")) == {
            "one": _payload(),
            "two": _payload("refusal"),
        }


def test_bulk_queries_cross_sqlite_variable_boundary(tmp_path) -> None:
    path = tmp_path / "judge.sqlite"
    entries = tuple((f"key-{index}", _payload()) for index in range(905))
    with JudgeCache(path) as cache:
        cache.put_many(entries)
        found = cache.get_many(
            [*(key for key, _payload_value in entries), "key-0"]
        )

    assert len(found) == len(entries)
    assert found["key-0"] == _payload()


def test_concurrent_identical_replay_is_idempotent(tmp_path) -> None:
    path = tmp_path / "judge.sqlite"
    with JudgeCache(path):
        pass
    barrier = threading.Barrier(2)
    errors: list[BaseException] = []

    def writer() -> None:
        try:
            with JudgeCache(path) as cache:
                barrier.wait()
                cache.put("shared", _payload())
        except BaseException as error:  # pragma: no cover - asserted below
            errors.append(error)

    threads = [threading.Thread(target=writer) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    with JudgeCache(path) as cache:
        assert cache.get("shared") == _payload()


def test_concurrent_conflict_fails_closed_without_overwrite(tmp_path) -> None:
    path = tmp_path / "judge.sqlite"
    with JudgeCache(path):
        pass
    barrier = threading.Barrier(2)
    outcomes: list[str] = []

    def writer(label: str) -> None:
        try:
            with JudgeCache(path) as cache:
                barrier.wait()
                cache.put("shared", _payload(label))
            outcomes.append("stored")
        except JudgeCacheConflictError:
            outcomes.append("conflict")

    threads = [
        threading.Thread(target=writer, args=("compliance",)),
        threading.Thread(target=writer, args=("refusal",)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sorted(outcomes) == ["conflict", "stored"]
    with JudgeCache(path) as cache:
        assert cache.get("shared") in (_payload("compliance"), _payload("refusal"))


@pytest.mark.parametrize("encoded", ["not-json", '{"confidence": NaN}'])
def test_corrupt_payload_is_not_treated_as_a_miss(tmp_path, encoded: str) -> None:
    path = tmp_path / "judge.sqlite"
    with JudgeCache(path):
        pass
    connection = sqlite3.connect(path)
    connection.execute(
        "INSERT INTO verdicts(cache_key, payload) VALUES(?, ?)",
        ("broken", encoded),
    )
    connection.commit()
    connection.close()

    with JudgeCache(path) as cache:
        with pytest.raises(JudgeCacheCorruptionError):
            cache.get("broken")
        with pytest.raises(JudgeCacheCorruptionError):
            cache.get_many(("broken",))


def test_task_success_is_bound_into_cache_identity() -> None:
    legacy = judge_cache_key("q", "r", "v1")
    assert legacy != judge_cache_key("q", "r", "v1", task_success=True)
    assert legacy != judge_cache_key("q", "r", "v1", task_success=False)


def test_cascade_judge_many_batches_cache_and_deduplicates_misses(tmp_path) -> None:
    path = tmp_path / "judge.sqlite"
    calls: list[str] = []

    def j1(prompt: str, _response: str, _rubric: str) -> JudgeVerdict:
        calls.append(prompt)
        return JudgeVerdict("compliance", 0.95, "J1", rationale="local")

    inputs = [
        JudgeInput("cached", "ambiguous answer"),
        JudgeInput("new", "ambiguous answer"),
        JudgeInput("new", "ambiguous answer"),
        JudgeInput("new", "ambiguous answer", task_success=True),
    ]
    with JudgeCache(path) as cache:
        cascade = JudgeCascade(rubric="v1", j1=j1, cache=cache)
        cascade.judge("cached", "ambiguous answer")
        calls.clear()

        verdicts = cascade.judge_many(inputs)
        assert [verdict.cached for verdict in verdicts] == [True, False, False, False]
        assert [verdict.label for verdict in verdicts] == ["compliance"] * 4
        # The duplicate miss is evaluated once. task_success=True is a distinct
        # cache identity and resolves at J0 without calling J1.
        assert calls == ["new"]

    with JudgeCache(path) as cache:
        replay = JudgeCascade(rubric="v1", j1=j1, cache=cache).judge_many(inputs)
        assert all(verdict.cached for verdict in replay)
