from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

gguf = pytest.importorskip("gguf")
from gguf import GGMLQuantizationType, GGUFReader, GGUFWriter  # noqa: E402
from gguf.quants import dequantize as python_dequantize  # noqa: E402

from heretic_nx.edits.gguf_codecs import (  # noqa: E402
    GGUFQuantizationCodecRegistry,
    NativeGGMLCodec,
    QUANT_LAYOUTS,
)
from heretic_nx.edits import gguf_quant as gguf_quant_module  # noqa: E402
from heretic_nx.edits.gguf_quant import (  # noqa: E402
    GGUFQuantizedAblationPlan,
    GGUFQuantizedTensorEdit,
    apply_quantized_gguf_ablation,
    inspect_quantized_gguf,
)
from heretic_nx.hashing import sha256_file  # noqa: E402


def _native_registry() -> GGUFQuantizationCodecRegistry:
    try:
        registry = GGUFQuantizationCodecRegistry()
        registry.ensure_supported(GGMLQuantizationType.Q4_K)
        return registry
    except RuntimeError as error:
        if os.environ.get("HERETIC_NX_GGML_LIBRARY"):
            pytest.fail(f"configured libggml-base is unusable: {error}")
        pytest.skip(f"libggml-base is unavailable: {error}")


def _write_quantized_gguf(
    path: Path,
    qtypes: tuple[GGMLQuantizationType, ...],
    *,
    expert_bank: bool = False,
) -> None:
    registry = (
        _native_registry()
        if any(QUANT_LAYOUTS[qtype.name].requires_native for qtype in qtypes)
        else GGUFQuantizationCodecRegistry(prefer_native=False)
    )
    writer = GGUFWriter(path, "llama")
    writer.add_name("heretic-nx-mixed-quant-test")
    for index, qtype in enumerate(qtypes):
        values = np.linspace(
            -2.0 + index * 0.1,
            2.0 + index * 0.1,
            (3 if expert_bank else 1) * 4 * 256,
            dtype=np.float32,
        ).reshape(((3, 4, 256) if expert_bank else (4, 256)))
        encoded = registry.quantize_rows(values.reshape(-1, 256), qtype).reshape(
            *values.shape[:-1], -1
        )
        writer.add_tensor(
            f"blk.{index}.attn_output.weight",
            encoded,
            raw_dtype=qtype,
        )
    writer.add_tensor("output_norm.weight", np.ones(4, dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def _write_plan(
    source: Path,
    factors: Path,
    plan: Path,
    *,
    qtype: str,
    strength: float = 0.7,
    target: str = "blk.0.attn_output.weight",
    preserve_original_blocks: bool = True,
    quantization_multipliers: tuple[float, ...] = (0.75, 1.0, 1.25),
) -> None:
    GGUFQuantizedAblationPlan(
        source_sha256=sha256_file(source),
        tensor_artifact_sha256=sha256_file(factors),
        row_chunk_size=2,
        edits=(
            GGUFQuantizedTensorEdit(
                tensor_name=target,
                expected_quantization=qtype,
                a_key="axis",
                strength=strength,
                require_payload_change=strength > 0,
                preserve_original_blocks=preserve_original_blocks,
                quantization_multipliers=quantization_multipliers,
            ),
        ),
    ).write(plan)


@pytest.mark.parametrize("qtype_name", ["Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K"])
def test_direct_k_quant_merge_preserves_layout_and_untouched_bytes(
    tmp_path: Path,
    qtype_name: str,
) -> None:
    source = tmp_path / "source.gguf"
    output = tmp_path / "edited.gguf"
    factors = tmp_path / "factors.safetensors"
    plan = tmp_path / "plan.json"
    qtype = getattr(GGMLQuantizationType, qtype_name)
    _write_quantized_gguf(source, (qtype, GGMLQuantizationType.Q8_0))
    save_file(
        {"axis": np.array([1.0, -0.5, 0.25, 0.1], dtype=np.float32)},
        factors,
    )
    _write_plan(source, factors, plan, qtype=qtype_name)

    report = apply_quantized_gguf_ablation(source, output, plan, factors)

    source_tensors = {tensor.name: tensor for tensor in GGUFReader(source).tensors}
    output_tensors = {tensor.name: tensor for tensor in GGUFReader(output).tensors}
    edited = output_tensors["blk.0.attn_output.weight"]
    assert edited.tensor_type == qtype
    assert edited.data.shape == source_tensors[edited.name].data.shape
    assert edited.n_bytes == source_tensors[edited.name].n_bytes
    assert np.array_equal(
        source_tensors["blk.1.attn_output.weight"].data,
        output_tensors["blk.1.attn_output.weight"].data,
    )
    row = report["edits"][0]
    assert row["payload_changed"]
    assert row["quantization_metrics"]["changed_blocks"] > 0
    assert row["quantization_metrics"]["delta_cosine"] > 0
    assert report["output"]["sha256"] == sha256_file(output)


@pytest.mark.parametrize("qtype_name", ["Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0"])
def test_common_quant_merge_is_transactional_without_native_codec(
    tmp_path: Path,
    qtype_name: str,
    monkeypatch,
) -> None:
    source = tmp_path / "source.gguf"
    output = tmp_path / "edited.gguf"
    factors = tmp_path / "factors.safetensors"
    plan = tmp_path / "plan.json"
    qtype = getattr(GGMLQuantizationType, qtype_name)
    _write_quantized_gguf(source, (qtype,))
    save_file(
        {"axis": np.array([1.0, -0.5, 0.25, 0.1], dtype=np.float32)},
        factors,
    )
    _write_plan(source, factors, plan, qtype=qtype_name)
    monkeypatch.setattr(
        "heretic_nx.edits.gguf_codecs._native_library_candidates",
        lambda _explicit: (),
    )

    report = apply_quantized_gguf_ablation(source, output, plan, factors)

    tensor = GGUFReader(output).tensors[0]
    assert tensor.tensor_type == qtype
    assert report["codec"]["backend"] == "gguf-python"
    assert report["edits"][0]["payload_changed"]


def test_zero_strength_is_bit_identical_for_k_quant(tmp_path: Path) -> None:
    source = tmp_path / "source.gguf"
    output = tmp_path / "edited.gguf"
    factors = tmp_path / "factors.safetensors"
    plan = tmp_path / "plan.json"
    _write_quantized_gguf(source, (GGMLQuantizationType.Q4_K,))
    save_file({"axis": np.ones(4, dtype=np.float32)}, factors)
    _write_plan(source, factors, plan, qtype="Q4_K", strength=0.0)

    report = apply_quantized_gguf_ablation(source, output, plan, factors)

    assert source.read_bytes() == output.read_bytes()
    assert not report["edits"][0]["payload_changed"]
    assert report["output"]["sha256"] == sha256_file(source)


def test_source_snapshot_rejects_hash_race(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source.gguf"
    output = tmp_path / "edited.gguf"
    factors = tmp_path / "factors.safetensors"
    plan = tmp_path / "plan.json"
    _write_quantized_gguf(source, (GGMLQuantizationType.Q4_K,))
    save_file({"axis": np.ones(4, dtype=np.float32)}, factors)
    _write_plan(source, factors, plan, qtype="Q4_K")
    original_copy = gguf_quant_module._copy_source

    def mutate_then_copy(source_path: Path, target_path: Path) -> None:
        with source_path.open("ab") as handle:
            handle.write(b"race")
        original_copy(source_path, target_path)

    monkeypatch.setattr(gguf_quant_module, "_copy_source", mutate_then_copy)

    with pytest.raises(RuntimeError, match="changed while.*snapshot"):
        apply_quantized_gguf_ablation(source, output, plan, factors)
    assert not output.exists()


def test_no_force_publish_preserves_concurrently_created_output(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "source.gguf"
    output = tmp_path / "edited.gguf"
    factors = tmp_path / "factors.safetensors"
    plan = tmp_path / "plan.json"
    _write_quantized_gguf(source, (GGMLQuantizationType.Q4_K,))
    save_file({"axis": np.ones(4, dtype=np.float32)}, factors)
    _write_plan(source, factors, plan, qtype="Q4_K")
    original_publish = gguf_quant_module._publish_output

    def racing_publish(temporary: Path, destination: Path, *, force: bool) -> str:
        destination.write_bytes(b"concurrent-writer")
        return original_publish(temporary, destination, force=force)

    monkeypatch.setattr(gguf_quant_module, "_publish_output", racing_publish)

    with pytest.raises(FileExistsError, match="concurrently-created"):
        apply_quantized_gguf_ablation(source, output, plan, factors)
    assert output.read_bytes() == b"concurrent-writer"


def test_min_drift_block_selection_never_worsens_target_error(tmp_path: Path) -> None:
    source = tmp_path / "source.gguf"
    min_drift_output = tmp_path / "min-drift.gguf"
    requantized_output = tmp_path / "requantized.gguf"
    factors = tmp_path / "factors.safetensors"
    min_drift_plan = tmp_path / "min-drift.json"
    requantized_plan = tmp_path / "requantized.json"
    _write_quantized_gguf(source, (GGMLQuantizationType.Q4_K,))
    save_file(
        {"axis": np.array([1.0, -0.5, 0.25, 0.1], dtype=np.float32)},
        factors,
    )
    _write_plan(
        source,
        factors,
        min_drift_plan,
        qtype="Q4_K",
        preserve_original_blocks=True,
        quantization_multipliers=(1.0,),
    )
    _write_plan(
        source,
        factors,
        requantized_plan,
        qtype="Q4_K",
        preserve_original_blocks=False,
        quantization_multipliers=(1.0,),
    )

    min_drift = apply_quantized_gguf_ablation(
        source, min_drift_output, min_drift_plan, factors
    )
    requantized = apply_quantized_gguf_ablation(
        source, requantized_output, requantized_plan, factors
    )
    selected = min_drift["edits"][0]["quantization_metrics"]
    candidate = requantized["edits"][0]["quantization_metrics"]

    assert (
        selected["target_approximation_rmse"]
        <= candidate["target_approximation_rmse"] + 1e-12
    )
    assert selected["changed_blocks"] <= candidate["changed_blocks"]


def test_mixed_quant_inspection_reports_real_tensor_types(tmp_path: Path) -> None:
    source = tmp_path / "mixed.gguf"
    _write_quantized_gguf(
        source,
        (
            GGMLQuantizationType.Q4_K,
            GGMLQuantizationType.Q6_K,
            GGMLQuantizationType.Q8_0,
        ),
    )

    report = inspect_quantized_gguf(source)

    assert report["quantization_histogram"] == {
        "F32": 1,
        "Q4_K": 1,
        "Q6_K": 1,
        "Q8_0": 1,
    }
    editable = {row["name"]: row for row in report["tensors"] if row["editable"]}
    assert editable["blk.0.attn_output.weight"]["quantization"] == "Q4_K"
    assert editable["blk.1.attn_output.weight"]["quantization"] == "Q6_K"
    assert "output_norm.weight" not in editable


def test_inspection_does_not_claim_k_editability_without_native_codec(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "mixed.gguf"
    _write_quantized_gguf(source, (GGMLQuantizationType.Q4_K,))
    monkeypatch.setattr(
        "heretic_nx.edits.gguf_codecs._native_library_candidates",
        lambda _explicit: (),
    )

    report = inspect_quantized_gguf(source)
    row = next(value for value in report["tensors"] if value["quantization"] == "Q4_K")

    assert not row["editable"]
    assert "codec unavailable" in row["ineligible_reasons"][0]
    assert not report["codec_availability"]["Q4_K"]["available"]


def test_chunked_k_quant_merge_handles_stacked_expert_banks(tmp_path: Path) -> None:
    source = tmp_path / "experts.gguf"
    output = tmp_path / "edited.gguf"
    factors = tmp_path / "factors.safetensors"
    plan = tmp_path / "plan.json"
    _write_quantized_gguf(source, (GGMLQuantizationType.Q4_K,), expert_bank=True)
    save_file(
        {"axis": np.array([1.0, -0.5, 0.25, 0.1], dtype=np.float32)},
        factors,
    )
    _write_plan(source, factors, plan, qtype="Q4_K")

    report = apply_quantized_gguf_ablation(source, output, plan, factors)

    row = report["edits"][0]
    assert row["matrix_count"] == 3
    assert row["quantization_metrics"]["total_blocks"] == 12
    assert (
        row["quantization_metrics"]["tracked_chunk_array_bytes_lower_bound"]
        < 32 * 1024
    )


def test_k_quant_plan_fails_closed_on_wrong_expected_type(tmp_path: Path) -> None:
    source = tmp_path / "source.gguf"
    factors = tmp_path / "factors.safetensors"
    plan = tmp_path / "plan.json"
    _write_quantized_gguf(source, (GGMLQuantizationType.Q4_K,))
    save_file({"axis": np.ones(4, dtype=np.float32)}, factors)
    _write_plan(source, factors, plan, qtype="Q6_K")

    with pytest.raises(RuntimeError, match="not Q6_K"):
        apply_quantized_gguf_ablation(source, None, plan, factors, dry_run=True)


def test_explicit_missing_native_library_fails_with_actionable_error(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="libggml-base"):
        NativeGGMLCodec(tmp_path / "missing-libggml-base.dylib")


def test_zero_strength_plan_requires_explicit_no_change_mode() -> None:
    with pytest.raises(ValueError, match="zero-strength"):
        GGUFQuantizedTensorEdit(
            tensor_name="blk.0.attn_output.weight",
            expected_quantization="Q4_K",
            a_key="axis",
            strength=0.0,
        )


def test_incomplete_native_library_fails_with_actionable_error(
    tmp_path: Path, monkeypatch
) -> None:
    library = tmp_path / "libggml-base.dylib"
    library.touch()
    monkeypatch.setattr("ctypes.CDLL", lambda _path: object())

    with pytest.raises(RuntimeError, match="missing required codec symbols"):
        NativeGGMLCodec(library)


def test_k_quant_ignores_portable_preference_and_uses_required_native_codec() -> None:
    registry = GGUFQuantizationCodecRegistry(prefer_native=False)
    try:
        registry.ensure_supported(GGMLQuantizationType.Q4_K)
    except RuntimeError as error:
        if os.environ.get("HERETIC_NX_GGML_LIBRARY"):
            pytest.fail(f"configured libggml-base is unusable: {error}")
        pytest.skip(f"libggml-base is unavailable: {error}")
    values = np.ones((2, 256), dtype=np.float32)

    encoded = registry.quantize_rows(values, GGMLQuantizationType.Q4_K)
    decoded = registry.dequantize_rows(encoded, GGMLQuantizationType.Q4_K, 256)

    assert registry.backend_for(GGMLQuantizationType.Q4_K) == "libggml-base"
    assert decoded.shape == values.shape


@pytest.mark.parametrize("qtype_name", ["Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0"])
def test_common_quant_codecs_have_portable_gguf_python_fallback(
    qtype_name: str,
) -> None:
    registry = GGUFQuantizationCodecRegistry(prefer_native=False)
    values = np.linspace(-3, 3, 5 * 64, dtype=np.float32).reshape(5, 64)
    qtype = getattr(GGMLQuantizationType, qtype_name)

    encoded = registry.quantize_rows(values, qtype)
    decoded = registry.dequantize_rows(encoded, qtype, 64)

    layout = QUANT_LAYOUTS[qtype_name]
    assert encoded.shape == (5, 64 // layout.block_size * layout.type_size)
    assert registry.backend_for(qtype) == "gguf-python"
    np.testing.assert_array_equal(
        decoded,
        python_dequantize(encoded, qtype),
    )


@pytest.mark.parametrize("qtype_name", ["Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K"])
def test_native_k_dequantization_matches_gguf_python(qtype_name: str) -> None:
    registry = _native_registry()
    qtype = getattr(GGMLQuantizationType, qtype_name)
    values = np.ascontiguousarray(
        np.random.default_rng(227).normal(size=(7, 512)), dtype=np.float32
    )

    encoded = registry.quantize_rows(values, qtype)
    native = registry.dequantize_rows(encoded, qtype, 512)

    np.testing.assert_array_equal(native, python_dequantize(encoded, qtype))
