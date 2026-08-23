from dataclasses import dataclass
from typing import Protocol


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

    tie_word_embeddings: bool
    gated_mlp: bool


class ModelFamily(Protocol):
    spec: ModelSpec

    def estimate_params(self) -> int: ...

    def calculate_kv_cache_bytes(
        self,
        context_length: int,
        bytes_per_token: int,
    ) -> int: ...
