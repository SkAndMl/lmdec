from transformers import AutoConfig

from lmdec.model_families import ModelSpec, resolve_family

from .presentation import render_analysis


def calculate_kv_bytes(
    num_kv_heads: int,
    head_dim: int,
    bytes_per_value: int,
    num_layers: int,
) -> int:
    total_num_values = num_layers * num_kv_heads * head_dim * 2
    return total_num_values * bytes_per_value


def estimate_params(model_spec: ModelSpec) -> int:

    if model_spec.gated_mlp:
        ffn_params = 3 * model_spec.hidden_size * model_spec.intermediate_size
    else:
        ffn_params = 2 * model_spec.hidden_size * model_spec.intermediate_size

    q_width = model_spec.num_attention_heads * model_spec.head_dim
    kv_width = model_spec.num_kv_heads * model_spec.head_dim

    qkv_params = (
        model_spec.hidden_size * q_width + 2 * model_spec.hidden_size * kv_width
    )
    output_proj_params = model_spec.hidden_size * model_spec.hidden_size

    embedding_params = model_spec.hidden_size * model_spec.vocab_size
    language_head_params = 0
    if not model_spec.tie_word_embeddings:
        language_head_params = model_spec.vocab_size * model_spec.hidden_size

    total_params = (
        embedding_params
        + (qkv_params + output_proj_params + ffn_params) * model_spec.num_layers
        + language_head_params
    )

    return total_params


def analyze(model_id: str) -> None:
    config = AutoConfig.from_pretrained(model_id)
    family = resolve_family(config)
    model_spec = family.build_spec(model_id, config)

    total_params = estimate_params(model_spec)

    kv_bytes_per_token = calculate_kv_bytes(
        num_kv_heads=model_spec.num_kv_heads,
        head_dim=model_spec.head_dim,
        bytes_per_value=2,
        num_layers=model_spec.num_layers,
    )

    print(render_analysis(model_spec, total_params, kv_bytes_per_token))
