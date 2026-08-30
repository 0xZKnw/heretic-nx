from __future__ import annotations

import hashlib
import os
import types
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
    _ResolvedEdit,
    _edit_tensor_payload,
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
    schema_version: str = "gguf-static-ablation-v3",
) -> None:
    GGUFQuantizedAblationPlan(
        schema_version=schema_version,
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


def _edited_payload_for_chunk_size(
    original: np.ndarray,
    factor: np.ndarray,
    qtype: GGMLQuantizationType,
    *,
    input_dim: int,
    row_chunk_size: int,
    arithmetic_mode: str = "chunk-stable-v1",
) -> tuple[np.ndarray, dict[str, object]]:
    registry = (
        _native_registry()
        if QUANT_LAYOUTS[qtype.name].requires_native
        else GGUFQuantizationCodecRegistry(prefer_native=False)
    )
    payload = original.copy()
    tensor = types.SimpleNamespace(
        tensor_type=qtype,
        shape=np.array([input_dim, payload.shape[0]]),
        data=payload,
        name="chunk-stable.weight",
    )
    edit = _ResolvedEdit(
        tensor_name=tensor.name,
        expected_quantization=qtype.name,
        a_key="axis",
        b_key=None,
        right_key=None,
        strength=0.7,
        preserve_row_norms=True,
        preserve_original_blocks=True,
        quantization_multipliers=(0.75, 1.0, 1.25),
        minimum_block_improvement=0.0,
        require_payload_change=True,
        minimum_delta_cosine=None,
        maximum_delta_relative_error=None,
        maximum_row_norm_relative_error=None,
    )
    try:
        report = _edit_tensor_payload(
            tensor,
            edit,
            {"axis": factor},
            registry,
            row_chunk_size=row_chunk_size,
            arithmetic_mode=arithmetic_mode,
        )
    finally:
        registry.close()
    return payload, report


@pytest.mark.parametrize("qtype_name", ["Q8_0", "Q4_K"])
def test_quantized_payload_is_independent_of_streaming_chunk_size(
    qtype_name: str,
) -> None:
    qtype = getattr(GGMLQuantizationType, qtype_name)
    registry = (
        _native_registry()
        if QUANT_LAYOUTS[qtype_name].requires_native
        else GGUFQuantizationCodecRegistry(prefer_native=False)
    )
    generator = np.random.default_rng(271)
    output_dim, input_dim, rank = 129, 512, 8
    values = np.ascontiguousarray(
        generator.normal(scale=0.08, size=(output_dim, input_dim)),
        dtype=np.float32,
    )
    factor = np.ascontiguousarray(
        generator.normal(size=(output_dim, rank)),
        dtype=np.float32,
    )
    factor /= np.linalg.norm(factor, axis=0, keepdims=True)
    try:
        original = registry.quantize_rows(values, qtype)
    finally:
        registry.close()

    results = [
        _edited_payload_for_chunk_size(
            original,
            factor,
            qtype,
            input_dim=input_dim,
            row_chunk_size=chunk_size,
        )
        for chunk_size in (1, 7, 64, 128)
    ]
    reference_payload, reference_report = results[0]
    for payload, report in results[1:]:
        np.testing.assert_array_equal(payload, reference_payload)
        assert (
            report["after_payload_sha256"]
            == reference_report["after_payload_sha256"]
        )
    legacy_payload, _ = _edited_payload_for_chunk_size(
        original,
        factor,
        qtype,
        input_dim=input_dim,
        row_chunk_size=128,
        arithmetic_mode="legacy-plan-v2",
    )
    np.testing.assert_array_equal(reference_payload, legacy_payload)


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
    assert report["schema_version"] == "gguf-quantized-static-merge-report-v3"
    assert report["plan"]["schema_version"] == "gguf-static-ablation-v3"
    assert report["arithmetic_mode"] == "chunk-stable-v1"


def test_v2_plan_replays_legacy_arithmetic_mode(tmp_path: Path) -> None:
    source = tmp_path / "source.gguf"
    factors = tmp_path / "factors.safetensors"
    plan = tmp_path / "plan.json"
    _write_quantized_gguf(source, (GGMLQuantizationType.Q8_0,))
    save_file({"axis": np.ones(4, dtype=np.float32)}, factors)
    _write_plan(
        source,
        factors,
        plan,
        qtype="Q8_0",
        schema_version="gguf-static-ablation-v2",
    )

    report = apply_quantized_gguf_ablation(
        source,
        None,
        plan,
        factors,
        dry_run=True,
    )

    assert report["plan"]["schema_version"] == "gguf-static-ablation-v2"
    assert report["arithmetic_mode"] == "legacy-plan-v2"


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

    def racing_publish(
        temporary: Path,
        destination: Path,
        *,
        force: bool,
        expected=None,
    ) -> str:
        destination.write_bytes(b"concurrent-writer")
        return original_publish(
            temporary,
            destination,
            force=force,
            expected=expected,
        )

    monkeypatch.setattr(gguf_quant_module, "_publish_output", racing_publish)

    with pytest.raises(FileExistsError, match="concurrently-created"):
        apply_quantized_gguf_ablation(source, output, plan, factors)
    assert output.read_bytes() == b"concurrent-writer"


def test_combined_file_hash_matches_independent_region_hashes(tmp_path: Path) -> None:
    path = tmp_path / "payload.bin"
    payload = bytes(range(251)) * 17
    path.write_bytes(payload)
    intervals = ((31, 217), (901, 1703))

    snapshot = gguf_quant_module._file_and_untouched_sha256(
        path,
        intervals,
        chunk_size=37,
    )
    untouched = payload[:31] + payload[217:901] + payload[1703:]

    assert snapshot.sha256 == hashlib.sha256(payload).hexdigest()
    assert snapshot.untouched_sha256 == hashlib.sha256(untouched).hexdigest()
    assert snapshot.size_bytes == len(payload)
    assert snapshot.inode == path.stat().st_ino


def test_publish_rejects_file_changed_after_final_hash(tmp_path: Path) -> None:
    temporary = tmp_path / "temporary.gguf"
    output = tmp_path / "output.gguf"
    temporary.write_bytes(b"before")
    snapshot = gguf_quant_module._file_and_untouched_sha256(temporary, ())
    temporary.write_bytes(b"after-with-a-different-size")

    with pytest.raises(RuntimeError, match="changed after final hashing"):
        gguf_quant_module._publish_output(
            temporary,
            output,
            force=False,
            expected=snapshot,
        )
    assert not output.exists()


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
