from transformers import PretrainedConfig

from lmdec.model_families.base import DType, ModelAnalysis, ModelSpec
from lmdec.model_families.helper import attention_type, dtype_to_byte_count


class Qwen2Family:
    def __init__(
        self,
        model_id: str,
        config: PretrainedConfig,
    ) -> None:

        head_dim = getattr(config, "head_dim", None)
        if head_dim is None:
            head_dim = config.hidden_size // config.num_attention_heads

        self.spec = ModelSpec(
            model_id=model_id,
            architecture=config.architectures[0],
            model_type=config.model_type,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=head_dim,
            attention_type=attention_type(
                config.num_key_value_heads, config.num_attention_heads
            ),
            vocab_size=config.vocab_size,
            context_window=config.max_position_embeddings,
            tie_word_embeddings=config.tie_word_embeddings,
            gated_mlp=True,
        )

    def estimate_params(self) -> int:
        if self.spec.gated_mlp:
            ffn_params = 3 * self.spec.hidden_size * self.spec.intermediate_size
        else:
            ffn_params = 2 * self.spec.hidden_size * self.spec.intermediate_size

        q_width = self.spec.head_dim * self.spec.num_attention_heads
        kv_width = self.spec.head_dim * self.spec.num_kv_heads

        qkv_params = (
            self.spec.hidden_size * q_width + 2 * self.spec.hidden_size * kv_width
        )
        output_proj_params = self.spec.hidden_size * self.spec.hidden_size

        embedding_params = self.spec.vocab_size * self.spec.hidden_size
        language_head_params = 0
        if not self.spec.tie_word_embeddings:
            language_head_params = self.spec.vocab_size * self.spec.hidden_size

        total_params = (
            embedding_params
            + self.spec.num_layers * (qkv_params + output_proj_params + ffn_params)
            + language_head_params
        )

        return total_params

    def calculate_kv_cache_bytes(
        self,
        bytes_per_value: int,
    ) -> int:

        total_value_per_token = (
            2 * self.spec.num_layers * self.spec.head_dim * self.spec.num_kv_heads
        )
        return bytes_per_value * total_value_per_token

    def analyze(
        self,
        context: int,
        batch_size: int,
        dtype: DType,
        kv_dtype: DType,
    ) -> ModelAnalysis:
        return ModelAnalysis(
            spec=self.spec,
            total_params=self.estimate_params(),
            kv_bytes_per_token=self.calculate_kv_cache_bytes(
                bytes_per_value=dtype_to_byte_count(kv_dtype),
            ),
            context=context,
            batch_size=batch_size,
            dtype=dtype,
            kv_dtype=kv_dtype,
        )
