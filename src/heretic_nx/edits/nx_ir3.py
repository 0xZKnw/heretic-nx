"""NX-IR3: evidence-bound static and runtime edit interchange format."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator

from heretic_nx.eval.promotion import GateEvidence, PrimeClaim, derive_prime_claim
from heretic_nx.hashing import canonical_json, sha256_file, sha256_json

from .nx_ir2 import SemanticSiteRef


SHA256_PATTERN = r"^[0-9a-f]{64}$"
CORE_ARTIFACTS = frozenset(
    {
        "base-model",
        "tokenizer",
        "chat-template",
        "config",
        "semantic-registry",
        "frozen-manifest",
        "tensors",
        "promotion-report",
    }
)


class ArtifactReferenceIR(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    sha256: str = Field(pattern=SHA256_PATTERN)


class GateEvidenceIR(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    gate: Literal[
        "geometry",
        "causal",
        "behavior",
        "capability",
        "provenance",
        "reproduction",
        "benchmark",
    ]
    status: Literal["pass", "fail", "not-run"]
    artifact_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_pass_artifact(self) -> "GateEvidenceIR":
        GateEvidence(self.gate, self.status, self.artifact_sha256, self.reason)
        return self


class EditIR3(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    site: SemanticSiteRef
    family: Literal[
        "directional",
        "leace-affine",
        "signed-spectral",
        "cayley",
        "low-rank-matrix",
        "atomic-unit",
    ]
    rank: int = Field(ge=1)
    beta: float = Field(ge=0, le=1)
    tensor_keys: tuple[str, ...] = Field(min_length=1)
    geometry_evidence_sha256: str = Field(pattern=SHA256_PATTERN)
    causal_evidence_sha256: str = Field(pattern=SHA256_PATTERN)
    route_id: str | None = None
    runtime_only: bool = False
    merge_targets: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_exportability(self) -> "EditIR3":
        if len(set(self.tensor_keys)) != len(self.tensor_keys):
            raise ValueError("edit tensor keys must be unique")
        if self.runtime_only and self.route_id is None:
            raise ValueError("runtime-only edits require a route")
        if not self.runtime_only and not self.merge_targets:
            raise ValueError("static edits require exact merge targets")
        return self


class NXIR3(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["3.0"] = "3.0"
    export_track: Literal["static-merge", "runtime-sidecar"]
    base_model_id: str
    base_model_revision: str
    artifacts: tuple[ArtifactReferenceIR, ...]
    split_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    evidence: tuple[GateEvidenceIR, ...]
    claim: PrimeClaim
    routes: tuple[str, ...] = ()
    edits: tuple[EditIR3, ...] = ()

    @model_validator(mode="after")
    def validate_integrity_chain(self) -> "NXIR3":
        artifacts = {artifact.name: artifact.sha256 for artifact in self.artifacts}
        if len(artifacts) != len(self.artifacts):
            raise ValueError("artifact names must be unique")
        missing = CORE_ARTIFACTS - set(artifacts)
        if missing:
            raise ValueError(f"NX-IR3 is missing core artifacts: {sorted(missing)}")
        evidence = {item.gate: item for item in self.evidence}
        if len(evidence) != len(self.evidence):
            raise ValueError("gate evidence entries must be unique")
        derived = derive_prime_claim(
            {
                name: GateEvidence(
                    item.gate,
                    item.status,
                    item.artifact_sha256,
                    item.reason,
                )
                for name, item in evidence.items()
            }
        )
        if self.claim != derived.claim:
            raise ValueError(
                f"declared claim {self.claim} is unsupported; evidence derives {derived.claim}"
            )
        route_ids = set(self.routes)
        if len(route_ids) != len(self.routes):
            raise ValueError("route ids must be unique")
        edit_ids = {edit.id for edit in self.edits}
        if len(edit_ids) != len(self.edits):
            raise ValueError("edit ids must be unique")
        for edit in self.edits:
            if edit.route_id is not None and edit.route_id not in route_ids:
                raise ValueError(f"edit {edit.id} references an unknown route")
            if self.export_track == "static-merge" and edit.runtime_only:
                raise ValueError("a static merge track cannot contain runtime-only edits")
            if self.export_track == "runtime-sidecar" and not edit.runtime_only:
                raise ValueError("a runtime sidecar cannot contain merge-only edits")
            for gate_name, digest in (
                ("geometry", edit.geometry_evidence_sha256),
                ("causal", edit.causal_evidence_sha256),
            ):
                gate = evidence.get(gate_name)
                if gate is None or gate.status != "pass" or gate.artifact_sha256 != digest:
                    raise ValueError(f"edit {edit.id} is not bound to passing {gate_name} evidence")
        return self

    @property
    def content_id(self) -> str:
        return sha256_json(self.model_dump())

    def verify_files(self, files: Mapping[str, str | Path]) -> None:
        """Verify every declared artifact before loading tensors or applying edits."""

        expected = {artifact.name: artifact.sha256 for artifact in self.artifacts}
        missing = sorted(set(expected) - set(files))
        if missing:
            raise RuntimeError(f"NX-IR3 artifact files are missing: {missing}")
        for name, digest in expected.items():
            path = Path(files[name])
            if not path.is_file():
                raise RuntimeError(f"NX-IR3 artifact {name} is not a file: {path}")
            actual = sha256_file(path)
            if actual != digest:
                raise RuntimeError(f"NX-IR3 artifact hash mismatch for {name}")

    def write(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(canonical_json(self.model_dump()) + b"\n")

    @classmethod
    def read(cls, path: str | Path) -> "NXIR3":
        return cls.model_validate_json(Path(path).read_bytes())
