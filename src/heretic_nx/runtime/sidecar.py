"""Verified loading of fail-closed NX-IR2 temporal sidecars."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from safetensors.torch import load_file
from torch import Tensor

from heretic_nx.edits.nx_ir2 import NXIR2, RoutePolicyIR, ThinkClosePolicyIR
from heretic_nx.hashing import sha256_file

from .latent_router import ConsensusSafetyRouter, LatentSafetyRouter, RouteDecision
from .temporal_logits import TemporalThinkController


@dataclass(frozen=True)
class LoadedTemporalSidecar:
    document: NXIR2
    route_policy: RoutePolicyIR
    generation_policy: ThinkClosePolicyIR
    router: ConsensusSafetyRouter

    @classmethod
    def load(
        cls,
        ir_path: str | Path,
        tensor_path: str | Path,
        *,
        allow_unaccepted: bool = False,
    ) -> "LoadedTemporalSidecar":
        document = NXIR2.read(ir_path)
        if document.accepted_report_sha256 is None and not allow_unaccepted:
            raise RuntimeError("NX-IR2 sidecar has no accepted promotion report")
        if sha256_file(tensor_path) != document.tensor_artifact_sha256:
            raise RuntimeError("NX-IR2 tensor artifact hash mismatch")
        if len(document.generation_controls) != 1:
            raise RuntimeError("exactly one temporal generation control is supported")
        generation = document.generation_controls[0]
        routes = {route.id: route for route in document.routes}
        route = routes[generation.route_id]
        if not route.risk_probes:
            raise RuntimeError("runtime temporal sidecars require semantic risk probe references")
        if not route.task_site_id or not route.task_labels:
            raise RuntimeError("task routing metadata is incomplete")
        tensors = load_file(tensor_path)
        required = {route.task_probe_key}
        for probe in route.risk_probes:
            required.update((probe.center_key, probe.scale_key, probe.axis_key))
        missing = required - set(tensors)
        if missing:
            raise RuntimeError(f"NX-IR2 tensor keys are missing: {sorted(missing)}")
        task_centroids = tensors[route.task_probe_key].float()
        probes = {}
        for probe in route.risk_probes:
            probes[probe.site_id] = LatentSafetyRouter(
                center=tensors[probe.center_key].float(),
                scale=tensors[probe.scale_key].float(),
                harmfulness_axis=tensors[probe.axis_key].float(),
                harmfulness_threshold=probe.threshold,
                task_labels=route.task_labels,
                task_centroids=task_centroids,
                minimum_task_similarity=route.task_threshold,
            )
        return cls(
            document,
            route,
            generation,
            ConsensusSafetyRouter(probes, task_site_id=route.task_site_id),
        )

    def decide(self, instruction_states: dict[str, Tensor]) -> RouteDecision:
        return self.router.decide(instruction_states)

    def make_controller(
        self,
        *,
        prompt_length: int,
        instruction_states: dict[str, Tensor],
    ) -> TemporalThinkController:
        decision = self.decide(instruction_states)
        return TemporalThinkController(
            prompt_length=prompt_length,
            open_token_id=self.generation_policy.open_token_id,
            close_token_id=self.generation_policy.close_token_id,
            budget_tokens=self.generation_policy.budget_tokens,
            grace_tokens=self.generation_policy.grace_tokens,
            close_logit_boost=self.generation_policy.close_logit_boost,
            risk_gate_passed=decision.action != "abstain-harmfulness",
            task_route_passed=decision.action == "route",
        )
