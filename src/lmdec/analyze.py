from typing import Any

from transformers import AutoConfig


def analyze(model_id: str):
    config: dict[str, Any] = AutoConfig.from_pretrained(model_id).to_dict()

    hidden_size: int = config["hidden_size"]
    num_attention_heads: int = config["num_attention_heads"]
    head_dim = hidden_size // num_attention_heads
    vocab_size: int = config["vocab_size"]
    num_layers: int = config["num_hidden_layers"]

    ffn_params: int = 3 * hidden_size * config["intermediate_size"]

    qkv_params: int = hidden_size * (config["num_attention_heads"] * head_dim) + 2 * (
        hidden_size * (config["num_key_value_heads"] * head_dim)
    )
    output_proj_params = hidden_size * hidden_size

    embedding_params = vocab_size * hidden_size

    language_head_params = 0
    if not config.get("tie_word_embeddings", False):
        language_head_params = vocab_size * hidden_size

    total_params = (
        embedding_params
        + (qkv_params + output_proj_params + ffn_params) * num_layers
        + language_head_params
    )
    total_million_params = total_params / 1_000_000

    print(model_id)
    print(f"Total params: ~{round(total_million_params, 2)}M")
