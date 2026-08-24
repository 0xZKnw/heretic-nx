"""Immutable, content-addressed experiment manifests."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .hashing import canonical_json, sha256_json


class FrozenManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = 1
    created_at_utc: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    model_id: str
    model_revision: str
    tokenizer_sha256: str
    chat_template_sha256: str
    config_sha256: str
    datasets: dict[str, str]
    seeds: tuple[int, ...]
    environment: dict[str, str]

    @property
    def content_id(self) -> str:
        payload = self.model_dump(exclude={"created_at_utc"})
        return sha256_json(payload)

    def write(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(canonical_json(self.model_dump()) + b"\n")


def read_manifest(path: str | Path) -> FrozenManifest:
    return FrozenManifest.model_validate_json(Path(path).read_bytes())


class ArtifactPin(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    revision: str
    sha256: str


class PrimeEnvironment(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    os_build: str
    python: str
    torch: str
    transformers: str
    bitsandbytes: str | None = None
    cuda_runtime: str | None = None
    cuda_driver: str | None = None
    execution_mode: str


class FrozenPrimeManifest(BaseModel):
    """PRIME manifest frozen before model loading or candidate selection."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: str = "2.0"
    created_at_utc: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    model: ArtifactPin
    tokenizer: ArtifactPin
    chat_template_sha256: str
    config_sha256: str
    datasets: tuple[ArtifactPin, ...]
    quantizers: dict[str, str]
    libraries: dict[str, str]
    seeds: tuple[int, ...]
    backend_mode: str
    environment: PrimeEnvironment
    semantic_registry_sha256: str
    golden_batch_sha256: str

    @property
    def content_id(self) -> str:
        return sha256_json(self.model_dump(exclude={"created_at_utc"}))

    def write(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(canonical_json(self.model_dump()) + b"\n")

    @classmethod
    def read(cls, path: str | Path) -> "FrozenPrimeManifest":
        return cls.model_validate_json(Path(path).read_bytes())
