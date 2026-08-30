from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from heretic_nx.eval.kl_integrity import (
    first_token_kl,
    load_completed_log_probabilities,
    load_completed_raw_logits,
    require_distinct_artifacts,
    require_matching_prompt_set,
    require_matching_runtime_protocol,
)
from heretic_nx.hashing import sha256_file


SCHEMA = "test-first-token-logprobs-v1"
RAW_SCHEMA = "test-first-token-raw-logits-v1"
COUNT = 2
VOCAB_SIZE = 3


def _progress(**updates: object) -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": SCHEMA,
        "label": "candidate",
        "model": "candidate.gguf",
        "prompt_tokens_sha256": "1" * 64,
        "artifact_sha256": "2" * 64,
        "runtime_model": {
            "endpoint": "http://127.0.0.1:1236",
            "model_alias": "candidate.gguf",
            "model_ftype": "Q8_0",
            "model_path": "/models/candidate.gguf",
            "artifact_sha256": "2" * 64,
            "artifact_size_bytes": 1024,
            "build_info": "b10621-test",
        },
        "vocab_size": VOCAB_SIZE,
        "count": COUNT,
        "completed": COUNT,
        "seconds": 1.25,
    }
    value.update(updates)
    return value


def _write_progress(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _valid_log_probabilities() -> np.ndarray:
    return np.log(
        np.array([[0.6, 0.3, 0.1], [0.2, 0.3, 0.5]], dtype=np.float32)
    ).astype(np.float32)


def test_load_completed_log_probabilities_accepts_valid_artifact(
    tmp_path: Path,
) -> None:
    array_path = tmp_path / "candidate.npy"
    progress_path = tmp_path / "candidate.progress.json"
    np.save(array_path, _valid_log_probabilities())
    _write_progress(progress_path, _progress())

    matrix, progress = load_completed_log_probabilities(
        array_path,
        progress_path,
        schema_version=SCHEMA,
        label="candidate",
        count=COUNT,
        vocab_size=VOCAB_SIZE,
    )

    np.testing.assert_array_equal(matrix, _valid_log_probabilities())
    assert progress["completed"] == COUNT


def test_first_token_kl_renormalizes_float32_rows() -> None:
    base = np.log(np.array([0.6, 0.3, 0.1], dtype=np.float32))
    candidate = np.log(np.array([0.5, 0.35, 0.15], dtype=np.float32))

    value = first_token_kl(base, candidate)

    expected = float(
        np.sum(
            np.array([0.6, 0.3, 0.1], dtype=np.float64)
            * np.log(
                np.array([0.6, 0.3, 0.1], dtype=np.float64)
                / np.array([0.5, 0.35, 0.15], dtype=np.float64)
            )
        )
    )
    assert value == pytest.approx(expected, abs=1e-7)
    assert first_token_kl(base, base) == pytest.approx(0.0, abs=1e-12)


def test_log_probability_artifact_rejects_incomplete_progress(
    tmp_path: Path,
) -> None:
    array_path = tmp_path / "candidate.npy"
    progress_path = tmp_path / "candidate.progress.json"
    np.save(array_path, _valid_log_probabilities())
    _write_progress(progress_path, _progress(completed=1))

    with pytest.raises(RuntimeError, match="completed"):
        load_completed_log_probabilities(
            array_path,
            progress_path,
            schema_version=SCHEMA,
            label="candidate",
            count=COUNT,
            vocab_size=VOCAB_SIZE,
        )


@pytest.mark.parametrize(
    ("matrix", "message"),
    [
        (np.zeros((COUNT, VOCAB_SIZE), dtype=np.float32), "distribution"),
        (
            np.array(
                [[np.nan, -1.0, -2.0], [-1.0, -1.0, -1.0]],
                dtype=np.float32,
            ),
            "non-finite",
        ),
        (np.zeros((1, VOCAB_SIZE), dtype=np.float32), "shape"),
        (_valid_log_probabilities().astype(np.float64), "dtype"),
    ],
)
def test_log_probability_artifact_rejects_invalid_matrix(
    tmp_path: Path, matrix: np.ndarray, message: str
) -> None:
    array_path = tmp_path / "candidate.npy"
    progress_path = tmp_path / "candidate.progress.json"
    np.save(array_path, matrix)
    _write_progress(progress_path, _progress())

    with pytest.raises(RuntimeError, match=message):
        load_completed_log_probabilities(
            array_path,
            progress_path,
            schema_version=SCHEMA,
            label="candidate",
            count=COUNT,
            vocab_size=VOCAB_SIZE,
        )


def test_progress_rejects_boolean_count_and_malformed_hash(tmp_path: Path) -> None:
    array_path = tmp_path / "candidate.npy"
    progress_path = tmp_path / "candidate.progress.json"
    np.save(array_path, _valid_log_probabilities())
    _write_progress(
        progress_path,
        _progress(count=True, prompt_tokens_sha256="not-a-hash"),
    )

    with pytest.raises(RuntimeError, match="count"):
        load_completed_log_probabilities(
            array_path,
            progress_path,
            schema_version=SCHEMA,
            label="candidate",
            count=COUNT,
            vocab_size=VOCAB_SIZE,
        )


def test_require_matching_prompt_set_rejects_misaligned_runs() -> None:
    base = _progress(label="base")
    candidate = _progress(prompt_tokens_sha256="3" * 64)

    with pytest.raises(RuntimeError, match="different prompt sets"):
        require_matching_prompt_set(
            base,
            candidate,
            base_path=Path("base.progress.json"),
            candidate_path=Path("candidate.progress.json"),
        )


def test_kl_comparison_requires_distinct_artifacts_and_matched_runtime() -> None:
    base = _progress(label="base")
    candidate = _progress(
        artifact_sha256="3" * 64,
        runtime_model={
            **base["runtime_model"],
            "artifact_sha256": "3" * 64,
            "model_path": "/models/candidate-edited.gguf",
        },
    )

    require_distinct_artifacts(base, candidate)
    require_matching_runtime_protocol(base, candidate)

    with pytest.raises(RuntimeError, match="identical SHA-256"):
        require_distinct_artifacts(base, base)
    mismatched = {
        **candidate,
        "runtime_model": {**candidate["runtime_model"], "build_info": "other"},
    }
    with pytest.raises(RuntimeError, match="build_info"):
        require_matching_runtime_protocol(base, mismatched)


def test_progress_requires_runtime_attestation_matching_artifact(
    tmp_path: Path,
) -> None:
    array_path = tmp_path / "candidate.npy"
    progress_path = tmp_path / "candidate.progress.json"
    np.save(array_path, _valid_log_probabilities())
    progress = _progress()
    progress["runtime_model"] = {
        **progress["runtime_model"],
        "artifact_sha256": "9" * 64,
    }
    _write_progress(progress_path, progress)

    with pytest.raises(RuntimeError, match="runtime artifact hash mismatch"):
        load_completed_log_probabilities(
            array_path,
            progress_path,
            schema_version=SCHEMA,
            label="candidate",
            count=COUNT,
            vocab_size=VOCAB_SIZE,
        )


def _raw_progress(path: Path, **updates: object) -> dict[str, object]:
    value = _progress(
        schema_version=RAW_SCHEMA,
        data_sha256=sha256_file(path),
    )
    value.update(updates)
    return value


def test_load_completed_raw_logits_requires_matching_content_hash(
    tmp_path: Path,
) -> None:
    data_path = tmp_path / "candidate.raw.bin"
    progress_path = tmp_path / "candidate.raw.progress.json"
    np.array([[1.0, 2.0, 3.0], [3.0, 1.0, -1.0]], dtype=np.float32).tofile(
        data_path
    )
    _write_progress(progress_path, _raw_progress(data_path, data_sha256="4" * 64))

    with pytest.raises(RuntimeError, match="hash does not match"):
        load_completed_raw_logits(
            data_path,
            progress_path,
            schema_version=RAW_SCHEMA,
            count=COUNT,
            vocab_size=VOCAB_SIZE,
        )


def test_load_completed_raw_logits_accepts_complete_artifact(
    tmp_path: Path,
) -> None:
    data_path = tmp_path / "candidate.raw.bin"
    progress_path = tmp_path / "candidate.raw.progress.json"
    expected = np.array(
        [[1.0, 2.0, 3.0], [3.0, 1.0, -1.0]], dtype=np.float32
    )
    expected.tofile(data_path)
    _write_progress(progress_path, _raw_progress(data_path))

    matrix, progress = load_completed_raw_logits(
        data_path,
        progress_path,
        schema_version=RAW_SCHEMA,
        count=COUNT,
        vocab_size=VOCAB_SIZE,
    )

    np.testing.assert_array_equal(matrix, expected)
    assert progress["data_sha256"] == sha256_file(data_path)


def test_load_completed_raw_logits_rejects_preallocated_zero_row(
    tmp_path: Path,
) -> None:
    data_path = tmp_path / "candidate.raw.bin"
    progress_path = tmp_path / "candidate.raw.progress.json"
    np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]], dtype=np.float32).tofile(
        data_path
    )
    _write_progress(progress_path, _raw_progress(data_path))

    with pytest.raises(RuntimeError, match="degenerate raw logits.*row 1"):
        load_completed_raw_logits(
            data_path,
            progress_path,
            schema_version=RAW_SCHEMA,
            count=COUNT,
            vocab_size=VOCAB_SIZE,
        )
