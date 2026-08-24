from transformers import PretrainedConfig

from lmdec.model_families.generic_decoder import GenericDecoderFamily
from lmdec.model_families.registry import resolve_family


class GenericConfig(PretrainedConfig):
    model_type = "generic-test"


def test_generic_decoder_builds_spec_from_common_aliases() -> None:
    config = GenericConfig(
        architectures=["GenericForCausalLM"],
        vocab_size=32_000,
        n_embd=768,
        n_layer=12,
        n_head=12,
        n_positions=2_048,
        n_inner=None,
        tie_word_embeddings=False,
    )

    family = GenericDecoderFamily("org/model", config)
    spec = family.spec

    assert spec.model_id == "org/model"
    assert spec.architecture == "GenericForCausalLM"
    assert spec.model_type == "generic-test"
    assert spec.hidden_size == 768
    assert spec.intermediate_size == 3_072
    assert spec.num_layers == 12
    assert spec.num_attention_heads == 12
    assert spec.num_kv_heads == 12
    assert spec.head_dim == 64
    assert spec.vocab_size == 32_000
    assert spec.context_window == 2_048
    assert spec.tie_word_embeddings is False
    assert spec.gated_mlp is False


def test_generic_decoder_respects_explicit_gated_mlp_config() -> None:
    config = GenericConfig(
        architectures=["GenericForCausalLM"],
        vocab_size=32_000,
        hidden_size=768,
        intermediate_size=3_072,
        num_hidden_layers=12,
        num_attention_heads=12,
        max_position_embeddings=2_048,
        gated_mlp=True,
    )

    family = GenericDecoderFamily("org/model", config)

    assert family.spec.gated_mlp is True


def test_generic_decoder_analysis_uses_requested_dtype_and_configuration() -> None:
    config = GenericConfig(
        architectures=["GenericForCausalLM"],
        vocab_size=32_000,
        hidden_size=768,
        intermediate_size=3_072,
        num_hidden_layers=12,
        num_attention_heads=12,
        max_position_embeddings=2_048,
    )
    family = GenericDecoderFamily("org/model", config)

    analysis = family.analyze(
        context=1_024,
        batch_size=4,
        dtype="fp32",
        kv_dtype="bf16",
    )

    assert analysis.spec is family.spec
    assert analysis.total_params == family.estimate_params()
    assert analysis.kv_bytes_per_token == 36 * 1_024
    assert analysis.context == 1_024
    assert analysis.batch_size == 4
    assert analysis.dtype == "fp32"
    assert analysis.kv_dtype == "bf16"


def test_registry_uses_generic_decoder_for_unknown_model_type() -> None:
    config = GenericConfig(
        architectures=["GenericForCausalLM"],
        vocab_size=32_000,
        n_embd=768,
        n_layer=12,
        n_head=12,
        n_positions=2_048,
    )

    family = resolve_family("org/model", config)

    assert isinstance(family, GenericDecoderFamily)
