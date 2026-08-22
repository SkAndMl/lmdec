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

    spec = GenericDecoderFamily().build_spec("org/model", config)

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
    assert spec.gated_mlp is True


def test_registry_uses_generic_decoder_for_unknown_model_type() -> None:
    family = resolve_family(GenericConfig())

    assert isinstance(family, GenericDecoderFamily)
