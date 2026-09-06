from transformers import PretrainedConfig

from lmdec.model_families.base import DType, GLM5Spec, ModelAnalysis
from lmdec.model_families.helper import dtype_to_byte_count


class GLM5Family:
    def __init__(
        self,
        model_id: str,
        config: PretrainedConfig,
    ) -> None:

        self.spec = GLM5Spec(
            model_id=model_id,
            model_type=config.model_type,
            architecture=config.architectures[0],
            vocab_size=config.vocab_size,
            context_window=config.max_position_embeddings,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            head_dim=config.head_dim,
            num_layers=config.num_hidden_layers,
            num_dense_layers=config.first_k_dense_replace,
            q_lora_rank=config.q_lora_rank,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            qk_head_dim=config.qk_nope_head_dim + config.qk_rope_head_dim,
            kv_lora_rank=config.kv_lora_rank,
            num_kv_heads=config.num_key_value_heads,
            v_head_dim=config.v_head_dim,
            intermediate_size=config.intermediate_size,
            moe_intermediate_size=config.moe_intermediate_size,
            num_routed_experts=config.n_routed_experts,
            num_shared_experts=config.n_shared_experts,
            num_activated_experts=config.num_experts_per_tok,
            index_head_dim=config.index_head_dim,
            index_n_heads=config.index_n_heads,
            index_topk=config.index_topk,
            tie_word_embeddings=config.tie_word_embeddings,
        )

    def estimate_params(self) -> int:

        embedding_params = self.spec.vocab_size * self.spec.hidden_size
        lm_head_params = 0
        if not self.spec.tie_word_embeddings:
            lm_head_params = self.spec.vocab_size * self.spec.hidden_size

        index_params = self._calculate_index_params()
        mla_params = self._calculate_mla_params()
        dense_mlp_params = self._calculate_dense_mlp_params()
        moe_mlp_params = self._calculate_moe_mlp_params()

        dense_block_params = index_params + mla_params + dense_mlp_params
        moe_block_params = index_params + mla_params + moe_mlp_params

        return (
            embedding_params
            + self.spec.num_dense_layers * dense_block_params
            + (self.spec.num_layers - self.spec.num_dense_layers) * moe_block_params
            + lm_head_params
        )

    def calculate_kv_cache_bytes(
        self,
        bytes_per_value: int,
    ) -> int:
        index_values = self.spec.index_head_dim * self.spec.num_layers
        mla_values = (
            self.spec.kv_lora_rank + self.spec.qk_rope_head_dim
        ) * self.spec.num_layers

        return (index_values + mla_values) * bytes_per_value

    def analyze(
        self,
        context: int,
        batch_size: int,
        dtype: DType,
        kv_dtype: DType,
        explain: bool,
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
            explain=explain,
        )

    def _calculate_index_params(self) -> int:

        wq_b_params = (
            self.spec.q_lora_rank * self.spec.index_n_heads * self.spec.index_head_dim
        )
        wk_params = self.spec.hidden_size * self.spec.index_head_dim
        weights_proj_params = self.spec.hidden_size * self.spec.index_n_heads

        return wq_b_params + wk_params + weights_proj_params

    def _calculate_mla_params(self) -> int:
        wq_a_params = self.spec.hidden_size * self.spec.q_lora_rank
        wq_b_params = (
            self.spec.q_lora_rank
            * self.spec.num_attention_heads
            * self.spec.qk_head_dim
        )

        wkv_a_params = self.spec.hidden_size * (
            self.spec.kv_lora_rank + self.spec.qk_rope_head_dim
        )
        wkv_b_params = (
            self.spec.kv_lora_rank
            * self.spec.num_attention_heads
            * (self.spec.qk_nope_head_dim + self.spec.v_head_dim)
        )

        wo_params = (
            self.spec.num_attention_heads * self.spec.v_head_dim * self.spec.hidden_size
        )

        return wq_a_params + wq_b_params + wkv_a_params + wkv_b_params + wo_params

    def _calculate_dense_mlp_params(self) -> int:
        return 3 * self.spec.intermediate_size * self.spec.hidden_size

    def _calculate_moe_mlp_params(self) -> int:
        expert_params = 3 * self.spec.hidden_size * self.spec.moe_intermediate_size
        router_params = self.spec.num_routed_experts * self.spec.hidden_size

        return (
            router_params
            + self.spec.num_routed_experts * expert_params
            + self.spec.num_shared_experts * expert_params
        )
