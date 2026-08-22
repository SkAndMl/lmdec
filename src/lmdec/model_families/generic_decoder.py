from typing import Any

from transformers import PretrainedConfig

from .base import ModelSpec


class GenericDecoderFamily:
    def build_spec(
        self,
        model_id: str,
        config: PretrainedConfig,
    ) -> ModelSpec:

        hidden_size = _required_attr(config, "hidden_size", "n_embd", "d_model")
        num_attention_heads = _required_attr(
            config,
            "num_attention_heads",
            "n_head",
        )

        intermediate_size = _optional_attr(
            config,
            "intermediate_size",
            "ffn_dim",
            "n_inner",
        )
        if intermediate_size is None:
            intermediate_size = 4 * hidden_size

        num_kv_heads = getattr(config, "num_key_value_heads", None)
        if num_kv_heads is None:
            num_kv_heads = num_attention_heads

        head_dim = getattr(config, "head_dim", None)
        if head_dim is None:
            head_dim = hidden_size // num_attention_heads

        architectures = getattr(config, "architectures", None)
        if architectures:
            architecture = architectures[0]
        else:
            architecture = config.__class__.__name__

        return ModelSpec(
            model_id=model_id,
            architecture=architecture,
            model_type=config.model_type,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_layers=_required_attr(
                config,
                "num_hidden_layers",
                "n_layer",
                "num_layers",
            ),
            num_attention_heads=num_attention_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            vocab_size=_required_attr(config, "vocab_size"),
            context_window=_required_attr(
                config,
                "max_position_embeddings",
                "n_positions",
                "seq_length",
            ),
            tie_word_embeddings=getattr(config, "tie_word_embeddings", True),
            gated_mlp=True,
        )


def _required_attr(config: PretrainedConfig, *names: str) -> Any:
    value = _optional_attr(config, *names)
    if value is None:
        names_text = ", ".join(names)
        raise ValueError(f"Configuration does not define any of: {names_text}")

    return value


def _optional_attr(config: PretrainedConfig, *names: str) -> Any | None:
    for name in names:
        value = getattr(config, name, None)
        if value is not None:
            return value

    return None
