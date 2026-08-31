"""No-download contracts for dense and non-attention sequence operators."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from heretic_nx.model.structural_frontend import (
    StructuralDiscoveryError,
    discover_structural_frontend,
    inspect_structural_frontend,
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


def _assert_sequence_operator_fails_closed(model: object) -> None:
    report = inspect_structural_frontend(model)  # type: ignore[arg-type]
    assert not report.is_unambiguous
    assert not report.targets_by_role("attention_output")
    assert not [
        site for site in report.activation_sites if site.role == "attention_output"
    ]
    assert any(
        item.code == "sequence_operator_unsupported" for item in report.ambiguities
    )
    with pytest.raises(StructuralDiscoveryError):
        discover_structural_frontend(model)  # type: ignore[arg-type]


def test_real_gptj_dense_outputs_are_both_discovered() -> None:
    model_class = _available("GPTJForCausalLM")
    config_class = _available("GPTJConfig")
    model = model_class(
        config_class(
            vocab_size=64,
            n_positions=32,
            n_embd=16,
            n_layer=1,
            n_head=4,
            rotary_dim=4,
            n_inner=32,
            pad_token_id=0,
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
    assert report.targets_by_role("attention_output")[0].parameter_path.endswith(
        "attn.out_proj.weight"
    )
    assert report.targets_by_role("ffn_output")[0].parameter_path.endswith(
        "mlp.fc_out.weight"
    )


def test_real_falcon_linear_outputs_are_both_discovered() -> None:
    model_class = _available("FalconForCausalLM")
    config_class = _available("FalconConfig")
    model = model_class(
        config_class(
            vocab_size=64,
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_kv_heads=2,
            max_position_embeddings=32,
            pad_token_id=0,
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
    assert report.targets_by_role("attention_output")[0].parameter_path.endswith(
        "self_attention.dense.weight"
    )
    assert report.targets_by_role("ffn_output")[0].parameter_path.endswith(
        "mlp.dense_4h_to_h.weight"
    )
    assert all(target.editable for target in report.weight_targets)


def test_real_gemma4_moe_router_is_not_a_residual_branch() -> None:
    model_class = _available("Gemma4ForCausalLM")
    config_class = _available("Gemma4TextConfig")
    model = model_class(
        config_class(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=4,
            max_position_embeddings=32,
            layer_types=["full_attention"],
            vocab_size_per_layer_input=64,
            hidden_size_per_layer_input=4,
            enable_moe_block=True,
            num_experts=2,
            top_k_experts=1,
            moe_intermediate_size=24,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
        )
    )
    inspected = inspect_structural_frontend(model)
    assert inspected.is_unambiguous
    assert not [item for item in inspected.ambiguities if item.path.endswith(".router")]
    report = discover_structural_frontend(model)
    assert {target.role for target in report.weight_targets} == {
        "attention_output",
        "ffn_output",
        "routed_ffn_output",
    }
    routed = report.targets_by_role("routed_ffn_output")[0]
    assert routed.parameter_path.endswith("experts.down_proj")
    assert not routed.editable


def test_real_gemma4_shared_kv_attention_is_discovered() -> None:
    model_class = _available("Gemma4ForCausalLM")
    config_class = _available("Gemma4TextConfig")
    model = model_class(
        config_class(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_kv_shared_layers=1,
            head_dim=4,
            max_position_embeddings=32,
            layer_types=["full_attention", "full_attention"],
            vocab_size_per_layer_input=64,
            hidden_size_per_layer_input=4,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
        )
    )

    report = discover_structural_frontend(model)

    attention_sites = tuple(
        site for site in report.activation_sites if site.role == "attention_output"
    )
    assert [site.attention_variant for site in attention_sites] == [
        "split",
        "shared_kv",
    ]
    assert all(
        target.editable
        for target in report.targets_by_role("attention_output")
    )


def _lfm2_model(*, layer_types: list[str]) -> object:
    model_class = _available("Lfm2ForCausalLM")
    config_class = _available("Lfm2Config")
    return model_class(
        config_class(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=len(layer_types),
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=32,
            block_multiple_of=8,
            block_auto_adjust_ff_dim=False,
            layer_types=layer_types,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
        )
    )


def test_real_lfm2_conv_is_not_mislabeled_as_attention() -> None:
    model = _lfm2_model(layer_types=["conv"])
    report = inspect_structural_frontend(model)  # type: ignore[arg-type]
    _assert_sequence_operator_fails_closed(model)
    operator = next(
        item
        for item in report.ambiguities
        if item.code == "sequence_operator_unsupported"
    )
    assert operator.layer == 0
    assert operator.path.endswith("layers.0.conv")


def test_real_lfm2_hybrid_only_labels_the_actual_attention_layer() -> None:
    model = _lfm2_model(layer_types=["conv", "full_attention"])
    report = inspect_structural_frontend(model)  # type: ignore[arg-type]
    attention_sites = [
        site for site in report.activation_sites if site.role == "attention_output"
    ]
    assert [(site.layer, site.module_path) for site in attention_sites] == [
        (1, "model.layers.1.self_attn.out_proj")
    ]
    assert any(
        item.code == "sequence_operator_unsupported"
        and item.layer == 0
        and item.path.endswith("layers.0.conv")
        for item in report.ambiguities
    )
    with pytest.raises(StructuralDiscoveryError):
        discover_structural_frontend(model)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("config_name", "model_name"),
    (
        ("MambaConfig", "MambaForCausalLM"),
        ("FalconMambaConfig", "FalconMambaForCausalLM"),
    ),
)
def test_real_state_space_mixer_is_not_mislabeled_as_attention(
    config_name: str,
    model_name: str,
) -> None:
    config_class: Callable[..., object] = _available(  # type: ignore[assignment]
        config_name
    )
    model_class: Callable[..., object] = _available(  # type: ignore[assignment]
        model_name
    )
    model = model_class(
        config_class(
            vocab_size=64,
            hidden_size=16,
            state_size=4,
            num_hidden_layers=1,
            expand=2,
            conv_kernel=3,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
        )
    )
    _assert_sequence_operator_fails_closed(model)
