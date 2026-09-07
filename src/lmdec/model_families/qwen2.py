from transformers import PretrainedConfig

from lmdec.model_families.attention import GroupedQueryAttention
from lmdec.model_families.base import BaseFamily, ModelSpec
from lmdec.model_families.mlp import DenseMLP


class Qwen2Family(BaseFamily):
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
            num_layers=config.num_hidden_layers,
            vocab_size=config.vocab_size,
            context_window=config.max_position_embeddings,
            tie_word_embeddings=config.tie_word_embeddings,
            attention=GroupedQueryAttention(
                num_attention_heads=config.num_attention_heads,
                num_kv_heads=config.num_key_value_heads,
                head_dim=head_dim,
            ),
            mlp=DenseMLP(intermediate_size=config.intermediate_size, gated=True),
        )
