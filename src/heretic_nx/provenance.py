"""DAG-structured, content-addressed provenance for PRIME runs."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator

from heretic_nx.hashing import canonical_json, sha256_directory, sha256_file, sha256_json


SHA256_PATTERN = r"^[0-9a-f]{64}$"
REQUIRED_PRIME_NODES = frozenset(
    {
        "base-model",
        "tokenizer",
        "chat-template",
        "config",
        "semantic-registry",
        "split-manifest",
        "engine-source",
        "response-artifact",
        "judge-evidence",
        "capability-report",
        "output-model",
        "promotion-report",
    }
)


class ProvenanceNode(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    kind: Literal["file", "directory"]
    sha256: str = Field(pattern=SHA256_PATTERN)
    depends_on: tuple[str, ...] = ()


class PrimeProvenanceManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["3.0"] = "3.0"
    run_id: str = Field(min_length=1)
    nodes: tuple[ProvenanceNode, ...]
    root: Literal["promotion-report"] = "promotion-report"

    @model_validator(mode="after")
    def validate_dag(self) -> "PrimeProvenanceManifest":
        by_name = {node.name: node for node in self.nodes}
        if len(by_name) != len(self.nodes):
            raise ValueError("provenance node names must be unique")
        missing = REQUIRED_PRIME_NODES - set(by_name)
        if missing:
            raise ValueError(f"provenance is missing required nodes: {sorted(missing)}")
        for node in self.nodes:
            unknown = set(node.depends_on) - set(by_name)
            if unknown:
                raise ValueError(f"node {node.name} has unknown dependencies: {sorted(unknown)}")
            if node.name in node.depends_on:
                raise ValueError(f"node {node.name} depends on itself")

        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(name: str) -> None:
            if name in visiting:
                raise ValueError("provenance graph contains a cycle")
            if name in visited:
                return
            visiting.add(name)
            for dependency in by_name[name].depends_on:
                visit(dependency)
            visiting.remove(name)
            visited.add(name)

        visit(self.root)
        unreachable = set(by_name) - visited
        if unreachable:
            raise ValueError(f"provenance nodes are not rooted in promotion-report: {sorted(unreachable)}")
        return self

    @property
    def content_id(self) -> str:
        return sha256_json(self.model_dump())

    def verify_paths(self, paths: Mapping[str, str | Path]) -> None:
        by_name = {node.name: node for node in self.nodes}
        missing = sorted(set(by_name) - set(paths))
        if missing:
            raise RuntimeError(f"provenance paths are missing: {missing}")
        for name, node in by_name.items():
            path = Path(paths[name])
            if node.kind == "file":
                if not path.is_file():
                    raise RuntimeError(f"provenance node {name} is not a file")
                actual = sha256_file(path)
            else:
                if not path.is_dir():
                    raise RuntimeError(f"provenance node {name} is not a directory")
                actual = sha256_directory(path)
            if actual != node.sha256:
                raise RuntimeError(f"provenance hash mismatch for {name}")

    def write(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(canonical_json(self.model_dump()) + b"\n")

    @classmethod
    def read(cls, path: str | Path) -> "PrimeProvenanceManifest":
        return cls.model_validate_json(Path(path).read_bytes())
