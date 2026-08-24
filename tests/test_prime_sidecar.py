from __future__ import annotations

import torch
from safetensors.torch import save_file

from heretic_nx.edits.nx_ir2 import (
    NXIR2,
    RiskProbeIR,
    RoutePolicyIR,
    ThinkClosePolicyIR,
)
from heretic_nx.hashing import sha256_file
from heretic_nx.runtime.sidecar import LoadedTemporalSidecar


def test_verified_temporal_sidecar_loads_and_routes_fail_closed(tmp_path) -> None:
    tensors = tmp_path / "router.safetensors"
    save_file(
        {
            "risk.center": torch.zeros(2),
            "risk.scale": torch.ones(2),
            "risk.axis": torch.tensor([1.0, 0.0]),
            "task.centroids": torch.tensor([[-1.0, 0.0]]),
        },
        tensors,
    )
    route = RoutePolicyIR(
        id="route",
        risk_probes=(
            RiskProbeIR(
                site_id="L12:block_out",
                center_key="risk.center",
                scale_key="risk.scale",
                axis_key="risk.axis",
                threshold=0.5,
            ),
        ),
        risk_aggregation="any",
        task_probe_key="task.centroids",
        task_site_id="L12:block_out",
        task_threshold=0.5,
        task_labels=("test",),
        calibration_sha256="9" * 64,
    )
    document = NXIR2(
        base_model_id="example/model",
        base_model_revision="abc",
        base_model_sha256="2" * 64,
        tokenizer_sha256="3" * 64,
        chat_template_sha256="4" * 64,
        frozen_manifest_sha256="5" * 64,
        semantic_registry_sha256="6" * 64,
        tensor_artifact_sha256=sha256_file(tensors),
        routes=(route,),
        generation_controls=(
            ThinkClosePolicyIR(
                id="close",
                route_id="route",
                open_token_id=7,
                close_token_id=8,
                budget_tokens=3,
                grace_tokens=1,
                close_logit_boost=5,
            ),
        ),
        accepted_report_sha256="a" * 64,
    )
    ir_path = tmp_path / "sidecar.json"
    document.write(ir_path)
    loaded = LoadedTemporalSidecar.load(ir_path, tensors)
    safe = {"L12:block_out": torch.tensor([-1.0, 0.0])}
    unsafe = {"L12:block_out": torch.tensor([1.0, 0.0])}
    assert loaded.decide(safe).action == "route"
    assert loaded.make_controller(prompt_length=2, instruction_states=safe).enabled
    assert not loaded.make_controller(prompt_length=2, instruction_states=unsafe).enabled

    draft = document.model_copy(update={"accepted_report_sha256": None})
    draft.write(ir_path)
    try:
        LoadedTemporalSidecar.load(ir_path, tensors)
    except RuntimeError as error:
        assert "no accepted promotion report" in str(error)
    else:
        raise AssertionError("draft sidecar must fail closed")
