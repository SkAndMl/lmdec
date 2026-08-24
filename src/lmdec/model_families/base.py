from dataclasses import dataclass
from typing import Literal, Protocol

DType = Literal["bf16", "fp16", "fp32"]


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    architecture: str
    model_type: str

    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_attention_heads: int
    num_kv_heads: int
    head_dim: int
    vocab_size: int
    context_window: int
    attention_type: Literal["MHA", "GQA", "MQA"]

    tie_word_embeddings: bool
    gated_mlp: bool


@dataclass(frozen=True)
class ModelAnalysis:
    spec: ModelSpec
    total_params: int
    kv_bytes_per_token: int
    context: int
    batch_size: int
    dtype: DType


class ModelFamily(Protocol):
    spec: ModelSpec

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
    ) -> ModelAnalysis: ...
