from transformers import Qwen2Config

from lmdec.model_families.qwen2 import Qwen2Family


def test_qwen2_family_calculates_head_dimension_when_not_explicit() -> None:
    config = Qwen2Config(
        architectures=["Qwen2ForCausalLM"],
        vocab_size=151_936,
        hidden_size=896,
        intermediate_size=4_864,
        num_hidden_layers=24,
        num_attention_heads=14,
        num_key_value_heads=2,
        max_position_embeddings=32_768,
        tie_word_embeddings=True,
    )

    spec = Qwen2Family().build_spec("Qwen/Qwen2.5-0.5B", config)

    assert spec.head_dim == 64
