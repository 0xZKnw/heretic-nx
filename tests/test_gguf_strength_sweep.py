from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

gguf = pytest.importorskip("gguf")
from gguf import GGMLQuantizationType, GGUFWriter  # noqa: E402

from heretic_nx.edits.gguf_codecs import (  # noqa: E402
    GGUFQuantizationCodecRegistry,
    QUANT_LAYOUTS,
)
from heretic_nx.edits.gguf_quant import (  # noqa: E402
    GGUFQuantizedAblationPlan,
    GGUFQuantizedTensorEdit,
    apply_quantized_gguf_ablation,
)
from heretic_nx.edits.gguf_sweep import (  # noqa: E402
    GGUFStrengthSweepCandidate,
    apply_quantized_gguf_strength_sweep,
)
from heretic_nx.hashing import sha256_file  # noqa: E402


def _registry_for(qtype: GGMLQuantizationType) -> GGUFQuantizationCodecRegistry:
    registry = GGUFQuantizationCodecRegistry(
        prefer_native=QUANT_LAYOUTS[qtype.name].requires_native
    )
    try:
        registry.ensure_supported(qtype)
    except RuntimeError as error:
        registry.close()
        if os.environ.get("HERETIC_NX_GGML_LIBRARY"):
            pytest.fail(f"configured libggml-base is unusable: {error}")
        pytest.skip(f"codec unavailable for {qtype.name}: {error}")
    return registry


def _write_source(
    path: Path,
    qtype: GGMLQuantizationType,
    *,
    expert_bank: bool = False,
    output_dim: int = 6,
    input_dim: int = 256,
    seed: int = 20260830,
) -> None:
    generator = np.random.default_rng(seed)
    shape = (3, output_dim, input_dim) if expert_bank else (output_dim, input_dim)
    values = np.ascontiguousarray(
        generator.normal(scale=0.18, size=shape), dtype=np.float32
    )
    registry = _registry_for(qtype)
    try:
        encoded = registry.quantize_rows(
            values.reshape(-1, input_dim), qtype
        ).reshape(*shape[:-1], -1)
    finally:
        registry.close()
    writer = GGUFWriter(path, "llama")
    writer.add_name("heretic-nx-strength-sweep-test")
    writer.add_tensor("blk.0.attn_output.weight", encoded, raw_dtype=qtype)
    writer.add_tensor("output_norm.weight", np.ones(output_dim, dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def _write_projector_factors(path: Path, *, output_dim: int = 6) -> None:
    generator = np.random.default_rng(17)
    axis = np.ascontiguousarray(
        generator.normal(size=(output_dim, 2)), dtype=np.float32
    )
    axis /= np.linalg.norm(axis, axis=0, keepdims=True)
    save_file({"axis": axis}, path)


def _write_direct_factors(
    path: Path,
    *,
    output_dim: int = 6,
    input_dim: int = 256,
) -> None:
    generator = np.random.default_rng(19)
    left = np.ascontiguousarray(
        generator.normal(scale=0.04, size=(output_dim, 2)), dtype=np.float32
    )
    right = np.ascontiguousarray(
        generator.normal(scale=0.04, size=(input_dim, 2)), dtype=np.float32
    )
    save_file({"left": left, "right": right}, path)


def _write_plan(
    source: Path,
    factors: Path,
    path: Path,
    *,
    qtype: str,
    strength: float,
    direct: bool = False,
    multiplier: float = 1.0,
    verify_untouched_bytes: bool = True,
) -> None:
    GGUFQuantizedAblationPlan(
        source_sha256=sha256_file(source),
        tensor_artifact_sha256=sha256_file(factors),
        row_chunk_size=2,
        verify_untouched_bytes=verify_untouched_bytes,
        edits=(
            GGUFQuantizedTensorEdit(
                tensor_name="blk.0.attn_output.weight",
                expected_quantization=qtype,
                a_key="left" if direct else "axis",
                right_key="right" if direct else None,
                strength=strength,
                preserve_row_norms=not direct,
                preserve_original_blocks=not direct,
                quantization_multipliers=(
                    (multiplier,)
                    if direct
                    else (0.75 * multiplier, multiplier, 1.25 * multiplier)
                ),
                require_payload_change=strength > 0,
            ),
        ),
    ).write(path)


def _run_exact_comparison(
    tmp_path: Path,
    qtype: GGMLQuantizationType,
    *,
    expert_bank: bool = False,
    direct: bool = False,
) -> dict[str, object]:
    source = tmp_path / "source.gguf"
    factors = tmp_path / "factors.safetensors"
    _write_source(source, qtype, expert_bank=expert_bank)
    if direct:
        _write_direct_factors(factors)
    else:
        _write_projector_factors(factors)
    strengths = (0.0, 0.35, 0.7, 1.05)
    plans: list[Path] = []
    sequential_reports = []
    candidates = []
    for index, strength in enumerate(strengths):
        plan = tmp_path / f"plan-{index}.json"
        sequential_output = tmp_path / f"sequential-{index}.gguf"
        sweep_output = tmp_path / f"sweep-{index}.gguf"
        _write_plan(
            source,
            factors,
            plan,
            qtype=qtype.name,
            strength=strength,
            direct=direct,
        )
        plans.append(plan)
        sequential_reports.append(
            apply_quantized_gguf_ablation(
                source, sequential_output, plan, factors
            )
        )
        candidates.append(
            GGUFStrengthSweepCandidate(
                label=f"beta-{strength:g}",
                plan_path=plan,
                output_path=sweep_output,
            )
        )

    sweep_report = apply_quantized_gguf_strength_sweep(
        source, factors, candidates
    )
    assert sweep_report["candidate_count"] == len(strengths)
    assert sweep_report["shared_source_dequantization"] is True
    assert sweep_report["shared_projector_reduction"] is (not direct)
    for index, candidate_row in enumerate(sweep_report["candidates"]):
        sequential_output = tmp_path / f"sequential-{index}.gguf"
        sweep_output = tmp_path / f"sweep-{index}.gguf"
        assert sha256_file(sweep_output) == sha256_file(sequential_output)
        merge_report = candidate_row["merge_report"]
        assert (
            merge_report["output"]["sha256"]
            == sequential_reports[index]["output"]["sha256"]
        )
        assert (
            merge_report["edits"][0]["after_payload_sha256"]
            == sequential_reports[index]["edits"][0]["after_payload_sha256"]
        )
        assert (
            merge_report["edits"][0]["quantization_metrics"]
            == sequential_reports[index]["edits"][0]["quantization_metrics"]
        )
    return sweep_report


@pytest.mark.parametrize(
    "qtype_name",
    ["Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q8_0"],
)
def test_strength_sweep_matches_independent_runs_bit_for_bit(
    tmp_path: Path, qtype_name: str
) -> None:
    _run_exact_comparison(
        tmp_path, getattr(GGMLQuantizationType, qtype_name)
    )


@pytest.mark.parametrize("qtype_name", ["Q4_0", "Q4_1", "Q5_0", "Q5_1"])
def test_strength_sweep_matches_portable_quantizers_bit_for_bit(
    tmp_path: Path, qtype_name: str
) -> None:
    _run_exact_comparison(
        tmp_path, getattr(GGMLQuantizationType, qtype_name)
    )


@pytest.mark.parametrize("qtype_name", ["Q4_K", "Q8_0"])
def test_strength_sweep_supports_stacked_moe_banks(
    tmp_path: Path, qtype_name: str
) -> None:
    _run_exact_comparison(
        tmp_path,
        getattr(GGMLQuantizationType, qtype_name),
        expert_bank=True,
    )


@pytest.mark.parametrize("qtype_name", ["Q2_K", "Q3_K", "Q5_K", "Q6_K"])
def test_strength_sweep_supports_low_bit_direct_moe_banks(
    tmp_path: Path, qtype_name: str
) -> None:
    _run_exact_comparison(
        tmp_path,
        getattr(GGMLQuantizationType, qtype_name),
        expert_bank=True,
        direct=True,
    )


@pytest.mark.parametrize("qtype_name", ["Q4_K", "Q8_0"])
def test_strength_sweep_matches_direct_right_factor_runs(
    tmp_path: Path, qtype_name: str
) -> None:
    _run_exact_comparison(
        tmp_path,
        getattr(GGMLQuantizationType, qtype_name),
        direct=True,
    )


def test_strength_sweep_matches_mixed_quant_multisite_runs(tmp_path: Path) -> None:
    source = tmp_path / "mixed.gguf"
    factors = tmp_path / "factors.safetensors"
    generator = np.random.default_rng(41)
    registry = _registry_for(GGMLQuantizationType.Q2_K)
    try:
        values = [
            np.ascontiguousarray(
                generator.normal(scale=0.12, size=(6, 256)), dtype=np.float32
            )
            for _ in range(2)
        ]
        encoded = (
            registry.quantize_rows(values[0], GGMLQuantizationType.Q2_K),
            registry.quantize_rows(values[1], GGMLQuantizationType.Q6_K),
        )
    finally:
        registry.close()
    writer = GGUFWriter(source, "llama")
    writer.add_name("heretic-nx-mixed-strength-sweep-test")
    writer.add_tensor(
        "blk.0.attn_output.weight",
        encoded[0],
        raw_dtype=GGMLQuantizationType.Q2_K,
    )
    writer.add_tensor(
        "blk.1.ffn_down.weight",
        encoded[1],
        raw_dtype=GGMLQuantizationType.Q6_K,
    )
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    axes = {
        "axis0": np.ascontiguousarray(
            generator.normal(scale=2.0, size=(6, 2)), dtype=np.float32
        ),
        "left1": np.ascontiguousarray(
            generator.normal(scale=0.04, size=(6, 2)), dtype=np.float32
        ),
        "right1": np.ascontiguousarray(
            generator.normal(scale=0.04, size=(256, 2)), dtype=np.float32
        ),
    }
    save_file(axes, factors)

    candidates = []
    sequential = []
    for index, strength in enumerate((0.4, 0.7, 1.0)):
        plan = tmp_path / f"mixed-{index}.json"
        GGUFQuantizedAblationPlan(
            source_sha256=sha256_file(source),
            tensor_artifact_sha256=sha256_file(factors),
            row_chunk_size=2,
            edits=(
                GGUFQuantizedTensorEdit(
                    tensor_name="blk.0.attn_output.weight",
                    expected_quantization="Q2_K",
                    a_key="axis0",
                    strength=strength,
                ),
                GGUFQuantizedTensorEdit(
                    tensor_name="blk.1.ffn_down.weight",
                    expected_quantization="Q6_K",
                    a_key="left1",
                    right_key="right1",
                    strength=strength * 0.6,
                    preserve_row_norms=False,
                    preserve_original_blocks=False,
                ),
            ),
        ).write(plan)
        independent = tmp_path / f"independent-{index}.gguf"
        sequential.append(
            apply_quantized_gguf_ablation(
                source, independent, plan, factors
            )
        )
        candidates.append(
            GGUFStrengthSweepCandidate(
                label=f"c{index}",
                plan_path=plan,
                output_path=tmp_path / f"sweep-{index}.gguf",
            )
        )
    report = apply_quantized_gguf_strength_sweep(source, factors, candidates)
    for index, candidate in enumerate(candidates):
        assert sha256_file(candidate.output_path) == sha256_file(
            tmp_path / f"independent-{index}.gguf"
        )
        assert (
            report["candidates"][index]["merge_report"]["output"]["sha256"]
            == sequential[index]["output"]["sha256"]
        )


def test_strength_sweep_rejects_non_strength_plan_difference(tmp_path: Path) -> None:
    source = tmp_path / "source.gguf"
    factors = tmp_path / "factors.safetensors"
    _write_source(source, GGMLQuantizationType.Q8_0)
    _write_projector_factors(factors)
    plans = (tmp_path / "plan-a.json", tmp_path / "plan-b.json")
    _write_plan(
        source, factors, plans[0], qtype="Q8_0", strength=0.4
    )
    _write_plan(
        source,
        factors,
        plans[1],
        qtype="Q8_0",
        strength=0.8,
        multiplier=0.9,
    )
    outputs = (tmp_path / "a.gguf", tmp_path / "b.gguf")
    candidates = [
        GGUFStrengthSweepCandidate(
            label=label, plan_path=plan, output_path=output
        )
        for label, plan, output in zip(("a", "b"), plans, outputs, strict=True)
    ]
    with pytest.raises(ValueError, match="differ only"):
        apply_quantized_gguf_strength_sweep(source, factors, candidates)
    assert not any(output.exists() for output in outputs)


def test_fast_search_sweep_matches_independent_search_runs(tmp_path: Path) -> None:
    source = tmp_path / "source.gguf"
    factors = tmp_path / "factors.safetensors"
    _write_source(source, GGMLQuantizationType.Q8_0)
    _write_direct_factors(factors)
    candidates = []
    sequential_reports = []
    for index, strength in enumerate((0.4, 0.8, 1.2)):
        plan = tmp_path / f"plan-{index}.json"
        sequential = tmp_path / f"sequential-{index}.gguf"
        _write_plan(
            source,
            factors,
            plan,
            qtype="Q8_0",
            strength=strength,
            direct=True,
            verify_untouched_bytes=False,
        )
        sequential_reports.append(
            apply_quantized_gguf_ablation(
                source,
                sequential,
                plan,
                factors,
                fast_search=True,
            )
        )
        candidates.append(
            GGUFStrengthSweepCandidate(
                label=f"c{index}",
                plan_path=plan,
                output_path=tmp_path / f"sweep-{index}.gguf",
            )
        )
    report = apply_quantized_gguf_strength_sweep(
        source, factors, candidates, fast_search=True
    )
    for index, candidate in enumerate(candidates):
        merge = report["candidates"][index]["merge_report"]
        assert sha256_file(candidate.output_path) == sha256_file(
            tmp_path / f"sequential-{index}.gguf"
        )
        assert merge["diagnostics_mode"] == "search"
        assert merge["untouched_bytes_verified"] is False
        assert "untouched_bytes_sha256" not in merge
        assert (
            merge["edits"][0]["quantization_metrics"]
            == sequential_reports[index]["edits"][0]["quantization_metrics"]
        )


def test_strength_sweep_rejects_source_race_without_publishing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from heretic_nx.edits import gguf_sweep as sweep_module

    source = tmp_path / "source.gguf"
    factors = tmp_path / "factors.safetensors"
    _write_source(source, GGMLQuantizationType.Q8_0)
    _write_projector_factors(factors)
    candidates = []
    for index, strength in enumerate((0.4, 0.8)):
        plan = tmp_path / f"plan-{index}.json"
        _write_plan(
            source, factors, plan, qtype="Q8_0", strength=strength
        )
        candidates.append(
            GGUFStrengthSweepCandidate(
                label=f"c{index}",
                plan_path=plan,
                output_path=tmp_path / f"out-{index}.gguf",
            )
        )

    original_copy = sweep_module._copy_source
    calls = 0

    def racing_copy(
        copy_source: Path,
        target: Path,
        *,
        minimum_free_after_copy: int = 0,
    ) -> str:
        nonlocal calls
        mode = original_copy(
            copy_source,
            target,
            minimum_free_after_copy=minimum_free_after_copy,
        )
        calls += 1
        if calls == 1:
            with copy_source.open("r+b") as handle:
                handle.seek(-1, os.SEEK_END)
                value = handle.read(1)
                handle.seek(-1, os.SEEK_END)
                handle.write(bytes((value[0] ^ 1,)))
                handle.flush()
                os.fsync(handle.fileno())
        return mode

    monkeypatch.setattr(sweep_module, "_copy_source", racing_copy)
    with pytest.raises(RuntimeError, match="changed while sweep snapshots"):
        apply_quantized_gguf_strength_sweep(source, factors, candidates)
    assert not any(candidate.output_path.exists() for candidate in candidates)


def test_strength_sweep_binds_reader_hash_and_copies_to_one_inode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An atomic pathname replacement cannot splice two source GGUFs."""

    from heretic_nx.edits import gguf_sweep as sweep_module

    source = tmp_path / "source.gguf"
    replacement = tmp_path / "replacement.gguf"
    factors = tmp_path / "factors.safetensors"
    _write_source(source, GGMLQuantizationType.Q8_0, seed=1)
    _write_source(replacement, GGMLQuantizationType.Q8_0, seed=2)
    _write_projector_factors(factors)
    candidates = []
    for index, strength in enumerate((0.4, 0.8)):
        plan = tmp_path / f"replacement-plan-{index}.json"
        # The plan deliberately names the replacement's content. Before the
        # fix, GGUFReader could retain source A while hashing/copying source B.
        _write_plan(
            replacement,
            factors,
            plan,
            qtype="Q8_0",
            strength=strength,
        )
        candidates.append(
            GGUFStrengthSweepCandidate(
                label=f"c{index}",
                plan_path=plan,
                output_path=tmp_path / f"bound-out-{index}.gguf",
            )
        )

    real_api = sweep_module._gguf_api()
    real_reader = real_api[0]
    calls = 0

    def racing_reader(path: Path, *args: object, **kwargs: object) -> object:
        nonlocal calls
        reader = real_reader(path, *args, **kwargs)
        calls += 1
        if calls == 1:
            os.replace(replacement, source)
        return reader

    monkeypatch.setattr(
        sweep_module,
        "_gguf_api",
        lambda: (racing_reader, *real_api[1:]),
    )
    with pytest.raises(RuntimeError, match="source GGUF hash mismatch"):
        apply_quantized_gguf_strength_sweep(source, factors, candidates)
    assert not any(candidate.output_path.exists() for candidate in candidates)


def test_all_zero_strength_sweep_reports_no_shared_arithmetic(tmp_path: Path) -> None:
    source = tmp_path / "source.gguf"
    factors = tmp_path / "factors.safetensors"
    _write_source(source, GGMLQuantizationType.Q8_0)
    _write_projector_factors(factors)
    candidates = []
    for index in range(2):
        plan = tmp_path / f"zero-{index}.json"
        _write_plan(source, factors, plan, qtype="Q8_0", strength=0.0)
        candidates.append(
            GGUFStrengthSweepCandidate(
                label=f"zero-{index}",
                plan_path=plan,
                output_path=tmp_path / f"zero-out-{index}.gguf",
            )
        )
    report = apply_quantized_gguf_strength_sweep(source, factors, candidates)
    assert report["shared_source_dequantization"] is False
    assert report["shared_projector_reduction"] is False
    assert set(report["snapshot_copy_modes"]) <= {"clone", "copy"}
    for candidate in candidates:
        assert sha256_file(candidate.output_path) == sha256_file(source)


def test_strength_sweep_api_is_publicly_imported() -> None:
    import heretic_nx.edits as edits

    assert edits.GGUFStrengthSweepCandidate is GGUFStrengthSweepCandidate
    assert (
        edits.apply_quantized_gguf_strength_sweep
        is apply_quantized_gguf_strength_sweep
    )


def test_copy_source_fallback_fails_before_unbudgeted_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from heretic_nx.edits import gguf_q8

    source = tmp_path / "payload.bin"
    target = tmp_path / "snapshot.bin"
    source.write_bytes(b"x" * 4096)
    target.touch()
    monkeypatch.setattr(gguf_q8.sys, "platform", "linux")
    disk_usage = type("usage", (), {"free": 8191})()
    monkeypatch.setattr(gguf_q8.shutil, "disk_usage", lambda _path: disk_usage)
    with pytest.raises(RuntimeError, match="full GGUF snapshot copy fallback"):
        gguf_q8._copy_source(
            source,
            target,
            minimum_free_after_copy=4096,
        )
    assert target.stat().st_size == 0


def test_strength_sweep_rolls_back_partial_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from heretic_nx.edits import gguf_sweep as sweep_module

    source = tmp_path / "source.gguf"
    factors = tmp_path / "factors.safetensors"
    _write_source(source, GGMLQuantizationType.Q8_0)
    _write_direct_factors(factors)
    candidates = []
    for index, strength in enumerate((0.4, 0.8, 1.2)):
        plan = tmp_path / f"plan-{index}.json"
        _write_plan(
            source,
            factors,
            plan,
            qtype="Q8_0",
            strength=strength,
            direct=True,
        )
        candidates.append(
            GGUFStrengthSweepCandidate(
                label=f"c{index}",
                plan_path=plan,
                output_path=tmp_path / f"out-{index}.gguf",
            )
        )

    original_publish = sweep_module._publish_output
    calls = 0

    def failing_publish(*args: object, **kwargs: object) -> str:
        nonlocal calls
        result = original_publish(*args, **kwargs)
        calls += 1
        if calls == 2:
            raise RuntimeError("injected post-publish failure")
        return result

    monkeypatch.setattr(sweep_module, "_publish_output", failing_publish)
    with pytest.raises(RuntimeError, match="injected post-publish"):
        apply_quantized_gguf_strength_sweep(source, factors, candidates)
    assert not any(candidate.output_path.exists() for candidate in candidates)
