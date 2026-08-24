from __future__ import annotations

from heretic_nx.eval.response_artifact import ResponseRecord, write_response_artifact
from heretic_nx.hashing import sha256_directory, sha256_file


def _record(item_id: str) -> ResponseRecord:
    return ResponseRecord(
        item_id=item_id,
        group_id=f"group-{item_id}",
        split="public-test",
        task="math",
        prompt="What is 2+2?",
        response="4",
        model_sha256="a" * 64,
        generation_config_sha256="b" * 64,
        judge_rubric_sha256="c" * 64,
        verdict="compliance",
        judge_level="J0",
        judge_confidence=1.0,
        rationale="exact answer",
        task_score=1.0,
    )


def test_response_artifact_is_complete_deterministic_and_immutable(tmp_path) -> None:
    path = tmp_path / "responses.jsonl"
    digest = write_response_artifact(path, [_record("a"), _record("b")])
    assert digest == sha256_file(path)
    assert len(path.read_text(encoding="utf-8").splitlines()) == 2
    try:
        write_response_artifact(path, [_record("c")])
    except FileExistsError:
        pass
    else:
        raise AssertionError("response evidence must not be overwritten")


def test_directory_hash_covers_relative_names_and_contents(tmp_path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "a.txt").write_text("same", encoding="utf-8")
    (second / "a.txt").write_text("same", encoding="utf-8")
    assert sha256_directory(first) == sha256_directory(second)
    (second / "b.txt").write_text("same", encoding="utf-8")
    assert sha256_directory(first) != sha256_directory(second)
