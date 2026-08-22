from transformers import PretrainedConfig

from .base import ModelSpec


class Qwen2Family:
    def build_spec(
        self,
        model_id: str,
        config: PretrainedConfig,
    ) -> ModelSpec:

        head_dim = (
            getattr(config, "head_dim", None)
            or config.hidden_size // config.num_attention_heads
        )

        return ModelSpec(
            model_id=model_id,
            architecture=config.architectures[0],
            model_type=config.model_type,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=head_dim,
            vocab_size=config.vocab_size,
            context_window=config.max_position_embeddings,
            tie_word_embeddings=config.tie_word_embeddings,
            gated_mlp=True,
        )
