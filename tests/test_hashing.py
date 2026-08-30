from __future__ import annotations

import math

import pytest

from heretic_nx.hashing import canonical_json, sha256_directory, sha256_file, sha256_json


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_canonical_json_rejects_non_finite_floats(value: float) -> None:
    with pytest.raises(ValueError, match="Out of range float values"):
        canonical_json({"nested": [0.0, value]})


def test_canonical_json_remains_deterministic_for_finite_values() -> None:
    assert canonical_json({"z": 1.5, "a": "café"}) == b'{"a":"caf\xc3\xa9","z":1.5}'


def test_directory_hash_is_exact_across_worker_counts(tmp_path) -> None:
    root = tmp_path / "model"
    (root / "nested").mkdir(parents=True)
    (root / "z.safetensors").write_bytes(b"z" * 257)
    (root / "a.json").write_bytes(b"configuration")
    (root / "nested" / "weights.bin").write_bytes(b"w" * 1025)

    files = sorted(
        (item for item in root.rglob("*") if item.is_file()),
        key=lambda item: item.relative_to(root).as_posix(),
    )
    expected = sha256_json(
        [
            {
                "path": file.relative_to(root).as_posix(),
                "sha256": sha256_file(file),
            }
            for file in files
        ]
    )

    assert sha256_directory(root, workers=1) == expected
    assert sha256_directory(root, workers=2) == expected
    assert sha256_directory(root, workers=8) == expected
    assert sha256_directory(root) == expected


@pytest.mark.parametrize("workers", [True, False, 0, -1, 1.5, "2"])
def test_directory_hash_rejects_invalid_worker_counts(tmp_path, workers: object) -> None:
    root = tmp_path / "model"
    root.mkdir()
    (root / "weights.bin").write_bytes(b"weights")

    with pytest.raises(ValueError, match="workers must be a positive integer or None"):
        sha256_directory(root, workers=workers)  # type: ignore[arg-type]
