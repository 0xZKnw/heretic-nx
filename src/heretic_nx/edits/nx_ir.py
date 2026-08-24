"""Backend-independent NX adapter manifest."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from heretic_nx.hashing import canonical_json, sha256_json


class ModuleEdit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    side: Literal["input", "output"]
    family: Literal["projector", "cayley", "lora"]
    rank: int = Field(ge=1)
    scale: float
    factor_keys: tuple[str, ...]
    protected_subspace_sha256: str

    @model_validator(mode="after")
    def validate_factors(self) -> "ModuleEdit":
        expected = 2
        if len(self.factor_keys) != expected:
            raise ValueError(f"{self.family} edits require exactly {expected} factor keys")
        return self


class NXIR(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    base_model_id: str
    base_model_revision: str
    base_model_sha256: str
    tokenizer_sha256: str
    chat_template_sha256: str
    calibration_manifest_sha256: str
    accepted_metrics_sha256: str | None = None
    modules: tuple[ModuleEdit, ...]

    @property
    def content_id(self) -> str:
        return sha256_json(self.model_dump())

    def write(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(canonical_json(self.model_dump()) + b"\n")

    @classmethod
    def read(cls, path: str | Path) -> "NXIR":
        return cls.model_validate_json(Path(path).read_bytes())
