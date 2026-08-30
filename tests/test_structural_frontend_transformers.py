"""Optional no-download contracts against real Transformers module layouts."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from heretic_nx.model.structural_frontend import (
    WeightLayout,
    discover_structural_frontend,
)


transformers = pytest.importorskip(
    "transformers",
    reason="real-layout contracts require the optional experiments dependency",
)


def _available(name: str) -> object:
    value = getattr(transformers, name, None)
    if value is None:
        pytest.skip(f"installed Transformers does not expose {name}")
    return value


@pytest.mark.parametrize(
    "model_factory",
    (
        lambda: _available("LlamaForCausalLM")(
            _available("LlamaConfig")(
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=4,
                num_key_value_heads=2,
                vocab_size=64,
                max_position_embeddings=32,
            )
        ),
        lambda: _available("GemmaForCausalLM")(
            _available("GemmaConfig")(
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=4,
                vocab_size=64,
                max_position_embeddings=32,
            )
        ),
    ),
    ids=("llama", "gemma"),
)
def test_real_dense_decoder_layouts(
    model_factory: Callable[[], object],
) -> None:
    report = discover_structural_frontend(model_factory())  # type: ignore[arg-type]
    assert report.decoder_stack_path == "model.layers"
    assert {target.role for target in report.weight_targets} == {
        "attention_output",
        "ffn_output",
    }
    assert {target.layout for target in report.weight_targets} == {WeightLayout.OI}
    assert all(target.editable for target in report.weight_targets)


def test_real_gpt2_conv1d_layout_is_io() -> None:
    model_class = _available("GPT2LMHeadModel")
    config_class = _available("GPT2Config")
    model = model_class(
        config_class(
            n_embd=16,
            n_layer=1,
            n_head=4,
            n_positions=32,
            n_ctx=32,
            vocab_size=64,
            bos_token_id=1,
            eos_token_id=2,
        )
    )
    report = discover_structural_frontend(model)
    assert report.decoder_stack_path == "transformer.h"
    assert {target.role for target in report.weight_targets} == {
        "attention_output",
        "ffn_output",
    }
    assert {target.layout for target in report.weight_targets} == {WeightLayout.IO}


def test_real_phi3_fused_gate_up_still_discovers_ffn_output() -> None:
    model_class = _available("Phi3ForCausalLM")
    config_class = _available("Phi3Config")
    model = model_class(
        config_class(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=64,
            max_position_embeddings=32,
            original_max_position_embeddings=32,
            pad_token_id=0,
        )
    )
    report = discover_structural_frontend(model)
    assert report.decoder_stack_path == "model.layers"
    assert {target.role for target in report.weight_targets} == {
        "attention_output",
        "ffn_output",
    }
    assert {target.layout for target in report.weight_targets} == {WeightLayout.OI}
    assert all(target.editable for target in report.weight_targets)


def test_real_mixtral_eoi_bank_is_described_but_not_editable() -> None:
    model_class = _available("MixtralForCausalLM")
    config_class = _available("MixtralConfig")
    model = model_class(
        config_class(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_local_experts=2,
            num_experts_per_tok=1,
            vocab_size=64,
            max_position_embeddings=32,
        )
    )
    report = discover_structural_frontend(model)
    routed = report.targets_by_role("routed_ffn_output")[0]
    assert routed.layout is WeightLayout.EOI
    assert routed.expert_count == 2
    assert routed.shape == (2, 16, 32)
    assert routed.activation_site_id is None
    assert not routed.editable
    assert any(item.code == "routed_ffn_edit_deferred" for item in report.exclusions)


def test_real_qwen_moe_separates_shared_and_routed_outputs() -> None:
    model_class = _available("Qwen2MoeForCausalLM")
    config_class = _available("Qwen2MoeConfig")
    model = model_class(
        config_class(
            hidden_size=16,
            intermediate_size=32,
            moe_intermediate_size=24,
            shared_expert_intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_experts=2,
            num_experts_per_tok=1,
            decoder_sparse_step=1,
            vocab_size=64,
            max_position_embeddings=32,
        )
    )
    report = discover_structural_frontend(model)
    shared = report.targets_by_role("shared_ffn_output")[0]
    routed = report.targets_by_role("routed_ffn_output")[0]
    assert shared.layout is WeightLayout.OI
    assert shared.editable
    assert shared.activation_site_id is not None
    assert routed.layout is WeightLayout.EOI
    assert routed.expert_count == 2
    assert not routed.editable
    assert routed.activation_site_id is None
