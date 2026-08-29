from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

gguf = pytest.importorskip("gguf")
from gguf import GGMLQuantizationType, GGUFReader, GGUFWriter  # noqa: E402
from gguf.quants import dequantize, quantize  # noqa: E402

from heretic_nx.edits.gguf_q8 import (  # noqa: E402
    GGUFQ8AblationPlan,
    GGUFQ8TensorEdit,
    apply_q8_gguf_ablation,
    inspect_q8_gguf,
)
from heretic_nx.hashing import sha256_file  # noqa: E402


def _write_tiny_gguf(path: Path) -> None:
    target = np.linspace(-2, 2, 128, dtype=np.float32).reshape(4, 32)
    untouched = np.linspace(-1, 1, 64, dtype=np.float32).reshape(2, 32)
    writer = GGUFWriter(path, "llama")
    writer.add_name("heretic-nx-q8-test")
    writer.add_tensor(
        "blk.0.attn_output.weight",
        quantize(target, GGMLQuantizationType.Q8_0),
        raw_dtype=GGMLQuantizationType.Q8_0,
    )
    writer.add_tensor(
        "blk.0.attn_q.weight",
        quantize(untouched, GGMLQuantizationType.Q8_0),
        raw_dtype=GGMLQuantizationType.Q8_0,
    )
    experts = np.linspace(-1, 1, 256, dtype=np.float32).reshape(2, 4, 32)
    writer.add_tensor(
        "blk.1.ffn_down_exps.weight",
        quantize(experts, GGMLQuantizationType.Q8_0),
        raw_dtype=GGMLQuantizationType.Q8_0,
    )
    writer.add_tensor("output_norm.weight", np.ones(4, dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def _write_plan(source: Path, factors: Path, plan: Path, *, target: str) -> None:
    document = GGUFQ8AblationPlan(
        source_sha256=sha256_file(source),
        tensor_artifact_sha256=sha256_file(factors),
        edits=(
            GGUFQ8TensorEdit(
                tensor_name=target,
                a_key="axis",
                strength=0.7,
            ),
        ),
    )
    document.write(plan)


def test_direct_q8_merge_changes_only_declared_payload(tmp_path: Path) -> None:
    source = tmp_path / "source.gguf"
    output = tmp_path / "edited.gguf"
    factors = tmp_path / "factors.safetensors"
    plan = tmp_path / "plan.json"
    _write_tiny_gguf(source)
    save_file({"axis": np.array([1.0, -0.5, 0.25, 0.1], dtype=np.float32)}, factors)
    _write_plan(source, factors, plan, target="blk.0.attn_output.weight")

    source_bytes = source.read_bytes()
    report = apply_q8_gguf_ablation(source, output, plan, factors)
    output_bytes = output.read_bytes()

    assert source.read_bytes() == source_bytes
    assert len(output_bytes) == len(source_bytes)
    source_reader = GGUFReader(source)
    output_reader = GGUFReader(output)
    source_tensors = {tensor.name: tensor for tensor in source_reader.tensors}
    output_tensors = {tensor.name: tensor for tensor in output_reader.tensors}
    target = source_tensors["blk.0.attn_output.weight"]
    start = target.data_offset
    stop = start + target.n_bytes
    assert source_bytes[:start] == output_bytes[:start]
    assert source_bytes[stop:] == output_bytes[stop:]
    assert not np.array_equal(target.data, output_tensors[target.name].data)
    assert np.array_equal(
        source_tensors["blk.0.attn_q.weight"].data,
        output_tensors["blk.0.attn_q.weight"].data,
    )

    original = dequantize(target.data, target.tensor_type)
    edited = dequantize(output_tensors[target.name].data, target.tensor_type)
    np.testing.assert_allclose(
        np.linalg.norm(edited, axis=1),
        np.linalg.norm(original, axis=1),
        rtol=0.02,
        atol=0.02,
    )
    assert report["edits"][0]["payload_changed"]
    assert report["output"]["sha256"] == sha256_file(output)


def test_q8_dry_run_validates_without_creating_output(tmp_path: Path) -> None:
    source = tmp_path / "source.gguf"
    factors = tmp_path / "factors.safetensors"
    plan = tmp_path / "plan.json"
    _write_tiny_gguf(source)
    save_file({"axis": np.ones(4, dtype=np.float32)}, factors)
    _write_plan(source, factors, plan, target="blk.0.attn_output.weight")

    report = apply_q8_gguf_ablation(
        source,
        None,
        plan,
        factors,
        dry_run=True,
    )

    assert report["dry_run"]
    assert report["edits"][0]["logical_shape"] == [4, 32]
    assert report["edits"][0]["quantization"] == "Q8_0"


def test_q8_merge_supports_stacked_moe_experts(tmp_path: Path) -> None:
    source = tmp_path / "source.gguf"
    output = tmp_path / "edited.gguf"
    factors = tmp_path / "factors.safetensors"
    plan = tmp_path / "plan.json"
    _write_tiny_gguf(source)
    save_file({"axis": np.array([1.0, -0.5, 0.25, 0.1], dtype=np.float32)}, factors)
    _write_plan(source, factors, plan, target="blk.1.ffn_down_exps.weight")

    report = apply_q8_gguf_ablation(source, output, plan, factors)

    source_tensor = next(
        tensor
        for tensor in GGUFReader(source).tensors
        if tensor.name == "blk.1.ffn_down_exps.weight"
    )
    output_tensor = next(
        tensor
        for tensor in GGUFReader(output).tensors
        if tensor.name == "blk.1.ffn_down_exps.weight"
    )
    assert report["edits"][0]["logical_shape"] == [2, 4, 32]
    assert report["edits"][0]["matrix_count"] == 2
    assert not np.array_equal(source_tensor.data, output_tensor.data)
    original = dequantize(source_tensor.data, source_tensor.tensor_type)
    edited = dequantize(output_tensor.data, output_tensor.tensor_type)
    np.testing.assert_allclose(
        np.linalg.norm(edited, axis=2),
        np.linalg.norm(original, axis=2),
        rtol=0.02,
        atol=0.02,
    )


def test_q8_merge_supports_direct_low_rank_delta(tmp_path: Path) -> None:
    source = tmp_path / "source.gguf"
    output = tmp_path / "edited.gguf"
    factors = tmp_path / "factors.safetensors"
    plan = tmp_path / "plan.json"
    _write_tiny_gguf(source)
    axis = np.array([1.0, -0.5, 0.25, 0.1], dtype=np.float32)
    right = np.linspace(-0.2, 0.3, 32, dtype=np.float32)
    save_file({"axis": axis, "right": right}, factors)
    GGUFQ8AblationPlan(
        source_sha256=sha256_file(source),
        tensor_artifact_sha256=sha256_file(factors),
        edits=(
            GGUFQ8TensorEdit(
                tensor_name="blk.0.attn_output.weight",
                a_key="axis",
                right_key="right",
                strength=0.7,
                preserve_row_norms=False,
            ),
        ),
    ).write(plan)

    report = apply_q8_gguf_ablation(source, output, plan, factors)

    source_tensor = next(
        tensor
        for tensor in GGUFReader(source).tensors
        if tensor.name == "blk.0.attn_output.weight"
    )
    output_tensor = next(
        tensor
        for tensor in GGUFReader(output).tensors
        if tensor.name == source_tensor.name
    )
    original = dequantize(source_tensor.data, source_tensor.tensor_type)
    edited = dequantize(output_tensor.data, output_tensor.tensor_type)
    expected = original - 0.7 * axis[:, None] @ right[None, :]
    np.testing.assert_allclose(edited, expected, rtol=0.03, atol=0.03)
    assert report["edits"][0]["edit_mode"] == "direct-low-rank"


def test_direct_low_rank_edit_rejects_projector_options() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        GGUFQ8TensorEdit(
            tensor_name="blk.0.attn_output.weight",
            a_key="axis",
            b_key="metric",
            right_key="right",
            strength=0.5,
            preserve_row_norms=False,
        )
    with pytest.raises(ValueError, match="cannot preserve row norms"):
        GGUFQ8TensorEdit(
            tensor_name="blk.0.attn_output.weight",
            a_key="axis",
            right_key="right",
            strength=0.5,
        )


def test_q8_merge_rejects_non_q8_target(tmp_path: Path) -> None:
    source = tmp_path / "source.gguf"
    factors = tmp_path / "factors.safetensors"
    plan = tmp_path / "plan.json"
    _write_tiny_gguf(source)
    save_file({"axis": np.ones(4, dtype=np.float32)}, factors)
    _write_plan(source, factors, plan, target="output_norm.weight")

    with pytest.raises(RuntimeError, match="not Q8_0"):
        apply_q8_gguf_ablation(source, None, plan, factors, dry_run=True)


def test_q8_plan_rejects_duplicate_targets() -> None:
    edit = GGUFQ8TensorEdit(
        tensor_name="blk.0.attn_output.weight",
        a_key="axis",
        strength=0.5,
    )
    with pytest.raises(ValueError, match="at most once"):
        GGUFQ8AblationPlan(
            source_sha256="1" * 64,
            tensor_artifact_sha256="2" * 64,
            edits=(edit, edit),
        )


def test_inspect_q8_gguf_lists_only_q8_tensors(tmp_path: Path) -> None:
    source = tmp_path / "source.gguf"
    _write_tiny_gguf(source)

    report = inspect_q8_gguf(source)

    assert report["tensor_count"] == 4
    assert report["q8_0_tensor_count"] == 3
    assert {row["name"] for row in report["q8_0_tensors"]} == {
        "blk.0.attn_output.weight",
        "blk.0.attn_q.weight",
        "blk.1.ffn_down_exps.weight",
    }
