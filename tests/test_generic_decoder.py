from transformers import PretrainedConfig

from lmdec.model_families.base import Workload
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
    assert spec.mlp.intermediate_size == 3_072
    assert spec.num_layers == 12
    assert spec.num_attention_heads == 12
    assert spec.attention.num_kv_heads == 12
    assert spec.attention.head_dim == 64
    assert spec.attention.attention_type == "MHA"
    assert spec.vocab_size == 32_000
    assert spec.context_window == 2_048
    assert spec.tie_word_embeddings is False
    assert spec.mlp.gated is False


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

    assert family.spec.mlp.gated is True


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
    workload = Workload(context=1_024, batch_size=4, dtype="fp32", kv_dtype="bf16")

    analysis = family.analyze(workload)

    assert analysis.spec is family.spec
    assert analysis.workload is workload
    assert analysis.params.total == family.spec.parameter_breakdown().total
    assert analysis.memory.kv_bytes_per_token == 36 * 1_024
    assert analysis.memory.kv_bytes_per_sequence == 36 * 1_024 * 1_024
    assert analysis.memory.kv_bytes_total == 36 * 1_024 * 1_024 * 4
    assert analysis.memory.weight_bytes == analysis.params.total * 4
    assert analysis.workload.context == 1_024
    assert analysis.workload.batch_size == 4
    assert analysis.workload.dtype == "fp32"
    assert analysis.workload.kv_dtype == "bf16"


def test_parameter_breakdown_components_sum_to_total() -> None:
    config = GenericConfig(
        architectures=["GenericForCausalLM"],
        vocab_size=32_000,
        hidden_size=768,
        intermediate_size=3_072,
        num_hidden_layers=12,
        num_attention_heads=12,
        max_position_embeddings=2_048,
        tie_word_embeddings=False,
    )
    spec = GenericDecoderFamily("org/model", config).spec

    params = spec.parameter_breakdown()

    assert params.embedding == 32_000 * 768
    assert params.lm_head == params.embedding
    assert params.mlp == 12 * 2 * 768 * 3_072
    assert (
        params.total
        == params.embedding + params.attention + params.mlp + params.lm_head
    )


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
