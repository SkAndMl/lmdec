from dataclasses import dataclass
from typing import Literal, Protocol

DType = Literal["bf16", "fp16", "fp32"]


@dataclass(frozen=True)
class BaseModelSpec:
    model_id: str
    architecture: str
    model_type: str

    hidden_size: int
    num_layers: int
    num_attention_heads: int

    vocab_size: int
    context_window: int

    tie_word_embeddings: bool


@dataclass(frozen=True)
class ModelSpec(BaseModelSpec):
    intermediate_size: int
    num_kv_heads: int
    head_dim: int
    attention_type: Literal["MHA", "GQA", "MQA", "DSA"]

    gated_mlp: bool


@dataclass(frozen=True)
class GLM5Spec(BaseModelSpec):
    num_dense_layers: int

    head_dim: int
    intermediate_size: int
    num_kv_heads: int

    ## indexer spec
    index_head_dim: int
    index_n_heads: int
    index_topk: int

    # mla spec
    q_lora_rank: int
    kv_lora_rank: int
    qk_head_dim: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int

    ## moe spec
    num_routed_experts: int
    num_shared_experts: int
    num_activated_experts: int
    moe_intermediate_size: int


@dataclass(frozen=True)
class ModelAnalysis:
    spec: BaseModelSpec
    total_params: int
    kv_bytes_per_token: int
    context: int
    batch_size: int
    dtype: DType
    kv_dtype: DType
    explain: bool


class ModelFamily(Protocol):
    @property
    def spec(self) -> BaseModelSpec: ...

    def estimate_params(self) -> int: ...

    def calculate_kv_cache_bytes(
        self,
        bytes_per_value: int,
    ) -> int: ...

    def analyze(
        self,
        context: int,
        batch_size: int,
        dtype: DType,
        kv_dtype: DType,
        explain: bool,
    ) -> ModelAnalysis: ...
