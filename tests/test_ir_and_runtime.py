from __future__ import annotations

from heretic_nx.edits.nx_ir import ModuleEdit, NXIR
from heretic_nx.runtime.memory_controller import AIMDMemoryController


def test_nx_ir_round_trip(tmp_path) -> None:
    document = NXIR(
        base_model_id="example/model",
        base_model_revision="abc123",
        base_model_sha256="0" * 64,
        tokenizer_sha256="1" * 64,
        chat_template_sha256="2" * 64,
        calibration_manifest_sha256="3" * 64,
        modules=(
            ModuleEdit(
                path="model.layers.4.self_attn.o_proj",
                side="output",
                family="projector",
                rank=2,
                scale=0.3,
                factor_keys=("layer4.a", "layer4.b"),
                protected_subspace_sha256="4" * 64,
            ),
        ),
    )
    path = tmp_path / "nx-ir.json"
    document.write(path)
    restored = NXIR.read(path)
    assert restored == document
    assert restored.content_id == document.content_id


def test_aimd_controller_is_operation_local_and_bounded() -> None:
    controller = AIMDMemoryController(batch_size=4, maximum_batch=8, stable_calls_before_growth=2)
    assert controller.observe(2 * 1024**3).batch_size == 4
    assert controller.observe(2 * 1024**3).batch_size == 5
    first_oom = controller.on_oom()
    assert first_oom.retry and first_oom.batch_size == 2
    controller.batch_size = 5
    second_oom_same_point = controller.on_oom()
    assert not second_oom_same_point.retry
