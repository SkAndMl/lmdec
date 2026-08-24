from typing import Any

from transformers import PretrainedConfig

from lmdec.model_families.base import ModelAnalysis, ModelSpec
from lmdec.model_families.helper import attention_type


class GenericDecoderFamily:
    def __init__(
        self,
        model_id: str,
        config: PretrainedConfig,
    ):

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

        self.spec = ModelSpec(
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
            attention_type=attention_type(num_kv_heads, num_attention_heads),
            vocab_size=_required_attr(config, "vocab_size"),
            context_window=_required_attr(
                config,
                "max_position_embeddings",
                "n_positions",
                "seq_length",
            ),
            tie_word_embeddings=getattr(config, "tie_word_embeddings", True),
            gated_mlp=getattr(config, "gated_mlp", False),
        )

    def calculate_kv_cache_bytes(
        self,
        context_length: int,
        bytes_per_value: int,
    ) -> int:
        total_value_per_token = (
            2 * self.spec.num_layers * self.spec.num_kv_heads * self.spec.head_dim
        )
        return context_length * bytes_per_value * total_value_per_token

    def estimate_params(self) -> int:

        if self.spec.gated_mlp:
            ffn_params = 3 * self.spec.hidden_size * self.spec.intermediate_size
        else:
            ffn_params = 2 * self.spec.hidden_size * self.spec.intermediate_size

        q_width = self.spec.num_attention_heads * self.spec.head_dim
        kv_width = self.spec.num_kv_heads * self.spec.head_dim

        qkv_params = (
            self.spec.hidden_size * q_width + 2 * self.spec.hidden_size * kv_width
        )
        output_proj_params = self.spec.hidden_size * self.spec.hidden_size

        embedding_params = self.spec.hidden_size * self.spec.vocab_size
        language_head_params = 0
        if not self.spec.tie_word_embeddings:
            language_head_params = self.spec.vocab_size * self.spec.hidden_size

        total_params = (
            embedding_params
            + (qkv_params + output_proj_params + ffn_params) * self.spec.num_layers
            + language_head_params
        )

        return total_params

    def analyze(self) -> ModelAnalysis:
        return ModelAnalysis(
            spec=self.spec,
            total_params=self.estimate_params(),
            kv_bytes_per_token=self.calculate_kv_cache_bytes(
                context_length=1,
                bytes_per_value=2,
            ),
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
