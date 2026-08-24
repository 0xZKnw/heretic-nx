"""Content-addressed, response-level evaluation artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field

from heretic_nx.hashing import canonical_json, sha256_file


class ResponseRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    item_id: str
    group_id: str
    split: Literal[
        "validation-search",
        "public-test",
        "secret-b",
        "secret-h",
        "cross-template",
        "cross-quant",
    ]
    task: str
    prompt: str
    response: str
    model_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    generation_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    judge_rubric_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    verdict: Literal["compliance", "refusal", "partial-refusal", "deflection", "ambiguous"]
    judge_level: Literal["J0", "J1", "J2", "J3-required", "J3"]
    judge_confidence: float = Field(ge=0, le=1)
    rationale: str
    task_score: float


def write_response_artifact(
    path: str | Path,
    records: Sequence[ResponseRecord],
    *,
    refuse_overwrite: bool = True,
) -> str:
    if not records:
        raise ValueError("at least one response record is required")
    keys = [(record.item_id, record.model_sha256) for record in records]
    if len(set(keys)) != len(keys):
        raise ValueError("response records must be unique per item and model")
    target = Path(path)
    if target.exists() and refuse_overwrite:
        raise FileExistsError(f"refusing to overwrite response artifact: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = b"".join(canonical_json(record.model_dump()) + b"\n" for record in records)
    target.write_bytes(payload)
    return sha256_file(target)
