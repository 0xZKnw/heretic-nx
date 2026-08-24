"""NX-IR 2 schema for canonical activation-native sidecars."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from heretic_nx.hashing import canonical_json, sha256_json


class SemanticSiteRef(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    layer: int = Field(ge=0)
    family: Literal["residual", "gqa", "ffn", "liv", "block"]
    kind: Literal["residual_out", "attention_out", "ffn_out", "liv_mix_out", "block_out"]
    module_path: str
    module_type: str
    stream_dim: int = Field(ge=1)
    structure_hash: str


class RiskProbeIR(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    site_id: str
    center_key: str
    scale_key: str
    axis_key: str
    threshold: float


class RoutePolicyIR(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    risk_probe_key: str | None = None
    task_probe_key: str
    risk_threshold: float | None = None
    task_threshold: float
    task_labels: tuple[str, ...] = ()
    calibration_sha256: str
    risk_probes: tuple[RiskProbeIR, ...] = ()
    risk_aggregation: Literal["single", "any"] = "single"
    task_site_id: str | None = None
    fail_closed: bool = True

    @model_validator(mode="after")
    def validate_probe_mode(self) -> "RoutePolicyIR":
        legacy = self.risk_probe_key is not None or self.risk_threshold is not None
        if legacy and (self.risk_probe_key is None or self.risk_threshold is None):
            raise ValueError("legacy risk probe key and threshold must be declared together")
        if bool(self.risk_probes) == legacy:
            raise ValueError("declare exactly one legacy risk probe or a probe ensemble")
        if self.risk_probes and self.risk_aggregation != "any":
            raise ValueError("multi-probe risk routing must use fail-closed any aggregation")
        return self


class TimePolicyIR(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    budget_tokens: int = Field(ge=1)
    maximum_window_tokens: int = Field(ge=1)
    checkpoint_stride: int = Field(ge=1)
    activation_threshold: float
    immediate_risk_shutdown: bool = True


class ThinkClosePolicyIR(BaseModel):
    """Fail-closed explicit-thinking generation controller."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    route_id: str
    open_token_id: int = Field(ge=0)
    close_token_id: int = Field(ge=0)
    budget_tokens: int = Field(ge=1)
    grace_tokens: int = Field(ge=0)
    close_logit_boost: float = Field(ge=0)
    force_after_grace: bool = True
    immediate_risk_shutdown: bool = True


class ActivationEditIR(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    site: SemanticSiteRef
    family: Literal["metric_projector", "cayley", "low_rank"]
    rank: int = Field(ge=1)
    beta: float = Field(ge=0, le=1)
    a_key: str
    b_key: str
    sparse_index_key: str | None = None
    route_id: str | None = None
    time_policy_id: str | None = None
    metric_sha256: str | None = None
    protected_subspace_sha256: str
    runtime_only: bool = True
    merge_targets: tuple[str, ...] = ()


class NXIR2(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: str = "2.0"
    base_model_id: str
    base_model_revision: str
    base_model_sha256: str
    tokenizer_sha256: str
    chat_template_sha256: str
    frozen_manifest_sha256: str
    semantic_registry_sha256: str
    tensor_artifact_sha256: str
    canonical_format: Literal["activation-sidecar"] = "activation-sidecar"
    routes: tuple[RoutePolicyIR, ...] = ()
    time_policies: tuple[TimePolicyIR, ...] = ()
    generation_controls: tuple[ThinkClosePolicyIR, ...] = ()
    edits: tuple[ActivationEditIR, ...] = ()
    accepted_report_sha256: str | None = None

    @model_validator(mode="after")
    def validate_references(self) -> "NXIR2":
        route_ids = {route.id for route in self.routes}
        time_ids = {policy.id for policy in self.time_policies}
        generation_ids = {policy.id for policy in self.generation_controls}
        if (
            len(route_ids) != len(self.routes)
            or len(time_ids) != len(self.time_policies)
            or len(generation_ids) != len(self.generation_controls)
        ):
            raise ValueError("route, time, and generation policy ids must be unique")
        if time_ids & generation_ids:
            raise ValueError("time and generation policy ids share one namespace")
        for policy in self.generation_controls:
            if policy.route_id not in route_ids:
                raise ValueError(f"unknown route id {policy.route_id}")
        for edit in self.edits:
            if edit.route_id is not None and edit.route_id not in route_ids:
                raise ValueError(f"unknown route id {edit.route_id}")
            if edit.time_policy_id is not None and edit.time_policy_id not in time_ids:
                raise ValueError(f"unknown time policy id {edit.time_policy_id}")
            if not edit.runtime_only and not edit.merge_targets:
                raise ValueError("mergeable edits must declare their exact merge targets")
        return self

    @property
    def content_id(self) -> str:
        return sha256_json(self.model_dump())

    def write(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(canonical_json(self.model_dump()) + b"\n")

    @classmethod
    def read(cls, path: str | Path) -> "NXIR2":
        return cls.model_validate_json(Path(path).read_bytes())
