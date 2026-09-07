from dataclasses import replace

from transformers import PretrainedConfig

from lmdec.model_families.attention import MultiHeadLatentAttention, SparseIndexer
from lmdec.model_families.base import BaseFamily, ModelSpec
from lmdec.model_families.mlp import DenseMLP, MixtureOfExperts


class DeepseekV3Family(BaseFamily):
    def __init__(
        self,
        model_id: str,
        config: PretrainedConfig,
    ) -> None:

        self.spec = ModelSpec(
            model_id=model_id,
            architecture=config.architectures[0],
            model_type=config.model_type,
            hidden_size=config.hidden_size,
            num_layers=config.num_hidden_layers,
            vocab_size=config.vocab_size,
            context_window=config.max_position_embeddings,
            tie_word_embeddings=config.tie_word_embeddings,
            attention=MultiHeadLatentAttention(
                num_attention_heads=config.num_attention_heads,
                q_lora_rank=config.q_lora_rank,
                kv_lora_rank=config.kv_lora_rank,
                qk_nope_head_dim=config.qk_nope_head_dim,
                qk_rope_head_dim=config.qk_rope_head_dim,
                v_head_dim=config.v_head_dim,
                indexer=None,
            ),
            mlp=MixtureOfExperts(
                dense=DenseMLP(intermediate_size=config.intermediate_size),
                num_dense_layers=config.first_k_dense_replace,
                expert_intermediate_size=config.moe_intermediate_size,
                num_routed_experts=config.n_routed_experts,
                num_shared_experts=config.n_shared_experts,
                num_activated_experts=config.num_experts_per_tok,
            ),
        )


class DeepseekV32Family(DeepseekV3Family):
    def __init__(
        self,
        model_id: str,
        config: PretrainedConfig,
    ) -> None:

        super().__init__(model_id, config)

        self.spec = replace(
            self.spec,
            attention=MultiHeadLatentAttention(
                num_attention_heads=config.num_attention_heads,
                q_lora_rank=config.q_lora_rank,
                kv_lora_rank=config.kv_lora_rank,
                qk_nope_head_dim=config.qk_nope_head_dim,
                qk_rope_head_dim=config.qk_rope_head_dim,
                v_head_dim=config.v_head_dim,
                indexer=SparseIndexer(
                    head_dim=config.index_head_dim,
                    num_heads=config.index_n_heads,
                    top_k=config.index_topk,
                ),
            ),
        )
