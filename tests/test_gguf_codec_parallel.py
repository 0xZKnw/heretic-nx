from __future__ import annotations

import os
import types

import numpy as np
import pytest


gguf = pytest.importorskip("gguf")
from gguf import GGMLQuantizationType  # noqa: E402

from heretic_nx.edits.gguf_codecs import (  # noqa: E402
    DEFAULT_PARALLEL_MIN_ELEMENTS,
    GGUFQuantizationCodecRegistry,
    NativeGGMLCodec,
    QUANT_LAYOUTS,
)
from heretic_nx.edits.gguf_quant import (  # noqa: E402
    _ResolvedEdit,
    _edit_tensor_payload,
)


def _native_codec(*, threads: int, parallel_min_elements: int = 1) -> NativeGGMLCodec:
    try:
        return NativeGGMLCodec(
            quantization_threads=threads,
            parallel_min_elements=parallel_min_elements,
        )
    except RuntimeError as error:
        if os.environ.get("HERETIC_NX_GGML_LIBRARY"):
            pytest.fail(f"configured libggml-base is unusable: {error}")
        pytest.skip(f"libggml-base is unavailable: {error}")


@pytest.mark.parametrize("qtype_name", sorted(QUANT_LAYOUTS))
def test_parallel_native_quantization_is_bit_identical(qtype_name: str) -> None:
    qtype = getattr(GGMLQuantizationType, qtype_name)
    values = np.ascontiguousarray(
        np.random.default_rng(251).normal(size=(129, 512)),
        dtype=np.float32,
    )

    with _native_codec(threads=1) as serial, _native_codec(threads=4) as parallel:
        serial_payload = serial.quantize_rows(values, qtype)
        parallel_payload = parallel.quantize_rows(values, qtype)
        serial_values = serial.dequantize_rows(serial_payload, qtype, values.shape[1])
        parallel_values = parallel.dequantize_rows(
            parallel_payload, qtype, values.shape[1]
        )

    np.testing.assert_array_equal(parallel_payload, serial_payload)
    np.testing.assert_array_equal(parallel_values, serial_values)


def test_parallel_threshold_keeps_small_quantization_serial() -> None:
    codec = _native_codec(
        threads=4,
        parallel_min_elements=DEFAULT_PARALLEL_MIN_ELEMENTS,
    )
    values = np.ones((4, 512), dtype=np.float32)

    try:
        codec.quantize_rows(values, GGMLQuantizationType.Q4_K)
        assert codec._executor is None
        codec.quantize_rows(
            np.ones((128, 512), dtype=np.float32),
            GGMLQuantizationType.Q4_K,
        )
        assert codec._executor is not None
    finally:
        codec.close()


@pytest.mark.parametrize("qtype_name", ["Q4_K", "Q6_K"])
def test_parallel_codec_preserves_complete_edit_payload_and_metrics(
    qtype_name: str,
) -> None:
    qtype = getattr(GGMLQuantizationType, qtype_name)
    generator = np.random.default_rng(257)
    output_dim, input_dim, rank = 128, 512, 4
    values = np.ascontiguousarray(
        generator.normal(scale=0.05, size=(output_dim, input_dim)),
        dtype=np.float32,
    )
    factor = np.ascontiguousarray(
        generator.normal(size=(output_dim, rank)), dtype=np.float32
    )
    factor /= np.linalg.norm(factor, axis=0, keepdims=True)
    edit = _ResolvedEdit(
        tensor_name="bench.weight",
        expected_quantization=qtype_name,
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

    with _native_codec(threads=1) as serial, _native_codec(threads=4) as parallel:
        original = serial.quantize_rows(values, qtype)
        serial_payload = original.copy()
        parallel_payload = original.copy()
        serial_tensor = types.SimpleNamespace(
            tensor_type=qtype,
            shape=np.array([input_dim, output_dim]),
            data=serial_payload,
            name="bench.weight",
        )
        parallel_tensor = types.SimpleNamespace(
            tensor_type=qtype,
            shape=np.array([input_dim, output_dim]),
            data=parallel_payload,
            name="bench.weight",
        )
        serial_report = _edit_tensor_payload(
            serial_tensor,
            edit,
            {"axis": factor},
            serial,
            row_chunk_size=64,
        )
        parallel_report = _edit_tensor_payload(
            parallel_tensor,
            edit,
            {"axis": factor},
            parallel,
            row_chunk_size=64,
        )

    np.testing.assert_array_equal(parallel_payload, serial_payload)
    assert parallel_report == serial_report


def test_parallel_configuration_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        GGUFQuantizationCodecRegistry(quantization_threads=0).ensure_supported(
            GGMLQuantizationType.Q4_K
        )
    with pytest.raises(ValueError, match="parallel_min_elements"):
        GGUFQuantizationCodecRegistry(parallel_min_elements=0)

    monkeypatch.setenv("HERETIC_NX_QUANT_THREADS", "invalid")
    with pytest.raises(ValueError, match="HERETIC_NX_QUANT_THREADS"):
        NativeGGMLCodec()


def test_native_provenance_records_parallel_execution_policy() -> None:
    with _native_codec(threads=3, parallel_min_elements=12345) as codec:
        provenance = codec.provenance()

    assert provenance["quantization_threads"] == 3
    assert provenance["parallel_min_elements"] == 12345
