"""Strict matched capability-report validation, independent of model libraries."""

from collections.abc import Mapping, Sequence
import math
from typing import Any

from heretic_nx.hashing import sha256_json


def validate_capability_pair(
    base: Mapping[str, Any], candidate: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    """Verify provenance, complete rows and recomputed scores before inference.

    Historical reports without an evidence digest are checked against the
    independently loaded frozen dataset. New reports also bind all their fields.
    """
    if not rows:
        raise ValueError("capability reference rows must be non-empty")
    if base["schema_version"] != candidate["schema_version"]:
        raise ValueError("capability schemas differ")
    if base["protocol"] != candidate["protocol"]:
        raise ValueError("capability protocols differ")
    for arm in (base, candidate):
        if arm["datasets"]["rows_sha256"] != sha256_json(rows):
            raise ValueError("capability rows do not match the frozen dataset")
        if arm["datasets"]["rows"] != len(rows) or arm["results"]["count"] != len(rows):
            raise ValueError("capability report is incomplete")
        digest = arm["artifact"]["sha256"]
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("capability artifact requires SHA-256")
        if arm["runtime"]["artifact_sha256"] != digest:
            raise ValueError("capability runtime artifact mismatch")
        predictions, correctness = arm["results"]["predictions"], arm["results"]["correctness"]
        if len(predictions) != len(rows) or len(correctness) != len(rows):
            raise ValueError("capability predictions are incomplete")
        if any(type(p) is not int or not 0 <= p < 4 for p in predictions):
            raise ValueError("invalid capability predictions")
        expected = [int(p == int(row["answer"])) for p, row in zip(predictions, rows, strict=True)]
        if any(type(x) is not int or x not in (0, 1) for x in correctness) or correctness != expected:
            raise ValueError("capability correctness does not match predictions")
        accuracy = arm["results"]["accuracy"]
        if not math.isfinite(accuracy) or abs(accuracy - sum(expected) / len(rows)) > 1e-12:
            raise ValueError("capability accuracy is inconsistent")
        tasks = {str(row["task"]) for row in rows}
        if set(arm["results"]["tasks"]) != tasks:
            raise ValueError("capability task slices differ")
        for task in tasks:
            selected = [expected[i] for i, row in enumerate(rows) if str(row["task"]) == task]
            observed = arm["results"]["tasks"][task]
            if observed["count"] != len(selected) or not math.isclose(observed["accuracy"], sum(selected) / len(selected), abs_tol=1e-12):
                raise ValueError("capability task score is inconsistent")
        if "evidence_sha256" in arm:
            payload = {k: v for k, v in arm.items() if k != "evidence_sha256"}
            if arm["evidence_sha256"] != sha256_json(payload):
                raise ValueError("capability evidence hash mismatch")
    # Model paths, aliases and endpoint ports may differ. The runtime build and
    # quantization must not; richer runtime attestations are compared if present.
    for key in ("build_info", "model_ftype", "runtime_protocol_sha256", "n_ctx", "n_batch", "n_ubatch"):
        if key in ("build_info", "model_ftype") and not base["runtime"].get(key):
            raise ValueError(f"missing capability runtime {key}")
        if base["runtime"].get(key) != candidate["runtime"].get(key):
            raise ValueError(f"capability runtime {key} differs")
