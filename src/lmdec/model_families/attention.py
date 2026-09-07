from dataclasses import dataclass
from math import gcd

from lmdec.model_families.base import DType
from lmdec.report import Row, Section


@dataclass(frozen=True)
class GroupedQueryAttention:
    num_attention_heads: int
    num_kv_heads: int
    head_dim: int

    def __post_init__(self) -> None:
        if self.num_attention_heads <= 0:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) must be positive"
            )
        if self.num_kv_heads <= 0:
            raise ValueError(f"num_kv_heads ({self.num_kv_heads}) must be positive")
        if self.num_kv_heads > self.num_attention_heads:
            raise ValueError(
                f"num_kv_heads ({self.num_kv_heads}) cannot exceed "
                f"num_attention_heads ({self.num_attention_heads})"
            )
        if self.head_dim <= 0:
            raise ValueError(f"head_dim ({self.head_dim}) must be positive")

    @property
    def attention_type(self) -> str:
        if self.num_kv_heads == self.num_attention_heads:
            return "MHA"
        if self.num_kv_heads == 1:
            return "MQA"
        return "GQA"

    @property
    def head_ratio(self) -> str:
        divisor = gcd(self.num_attention_heads, self.num_kv_heads)
        return f"{self.num_attention_heads // divisor}:{self.num_kv_heads // divisor}"

    def parameters(self, hidden_size: int, num_layers: int) -> int:
        query = hidden_size * self.num_attention_heads * self.head_dim
        key_value = 2 * hidden_size * self.num_kv_heads * self.head_dim
        output = hidden_size * hidden_size

        return num_layers * (query + key_value + output)

    def kv_bytes_per_token(self, num_layers: int, bytes_per_element: int) -> int:
        values = 2 * num_layers * self.num_kv_heads * self.head_dim
        return values * bytes_per_element

    def dimension_rows(self) -> list[Row]:
        return [
            Row("KV heads", f"{self.num_kv_heads:,}"),
            Row("Head dimension", f"{self.head_dim:,}"),
        ]

    def summary_rows(self) -> list[Row]:
        return [
            Row("Type", self.attention_type),
            Row("Query heads", f"{self.num_attention_heads:,}"),
            Row("KV heads", f"{self.num_kv_heads:,}"),
            Row("Q:KV ratio", self.head_ratio),
        ]

    def cache_layout_rows(self) -> list[Row]:
        return []

    def derivation_sections(self, hidden_size: int) -> list[Section]:
        return [
            Section(
                "HEAD DIMENSION",
                [
                    f"{hidden_size:,} / {self.num_attention_heads:,} query heads",
                    f"→ {self.head_dim:,}",
                ],
            ),
            Section(
                "ATTENTION",
                [
                    (
                        f"{self.num_attention_heads:,} query heads / "
                        f"{self.num_kv_heads:,} KV heads"
                    ),
                    f"→ {self.attention_type}, Q:KV ratio {self.head_ratio}",
                    *self._sharing_lines(),
                ],
            ),
        ]

    def cache_derivation_lines(self, num_layers: int, dtype: DType) -> list[str]:
        return [
            (
                f"2 (K + V) × {num_layers:,} layers × "
                f"{self.num_kv_heads:,} KV heads × {self.head_dim:,} × "
                f"{dtype.bytes_per_element} bytes ({dtype.label})"
            ),
        ]

    def assumptions(self) -> list[str]:
        return []

    def _sharing_lines(self) -> list[str]:
        if self.num_attention_heads == self.num_kv_heads:
            return ["→ Each query head has its own KV head; no KV-head sharing."]

        if self.num_attention_heads % self.num_kv_heads == 0:
            sharing_factor = self.num_attention_heads // self.num_kv_heads
            return [
                (
                    f"→ Each KV head is shared by {sharing_factor:,} query heads, "
                    "reducing KV-cache"
                ),
                f"  payload by {sharing_factor:,}× versus equivalent MHA.",
            ]

        return ["→ Fewer KV heads than query heads reduce KV-cache payload versus MHA."]


@dataclass(frozen=True)
class SparseIndexer:
    head_dim: int
    num_heads: int
    top_k: int

    def parameters(self, hidden_size: int, q_lora_rank: int) -> int:
        query = q_lora_rank * self.num_heads * self.head_dim
        key = hidden_size * self.head_dim
        weights = hidden_size * self.num_heads

        return query + key + weights

    def values_per_token(self) -> int:
        return self.head_dim


@dataclass(frozen=True)
class MultiHeadLatentAttention:
    num_attention_heads: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    indexer: SparseIndexer | None = None

    def __post_init__(self) -> None:
        if self.num_attention_heads <= 0:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) must be positive"
            )
        if self.kv_lora_rank <= 0:
            raise ValueError(f"kv_lora_rank ({self.kv_lora_rank}) must be positive")

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    @property
    def attention_type(self) -> str:
        return "DSA (MLA)" if self.indexer is not None else "MLA"

    def parameters(self, hidden_size: int, num_layers: int) -> int:
        query_down = hidden_size * self.q_lora_rank
        query_up = self.q_lora_rank * self.num_attention_heads * self.qk_head_dim

        key_value_down = hidden_size * (self.kv_lora_rank + self.qk_rope_head_dim)
        key_value_up = (
            self.kv_lora_rank
            * self.num_attention_heads
            * (self.qk_nope_head_dim + self.v_head_dim)
        )

        output = self.num_attention_heads * self.v_head_dim * hidden_size

        per_layer = query_down + query_up + key_value_down + key_value_up + output
        if self.indexer is not None:
            per_layer += self.indexer.parameters(hidden_size, self.q_lora_rank)

        return num_layers * per_layer

    def kv_bytes_per_token(self, num_layers: int, bytes_per_element: int) -> int:
        values = self.kv_lora_rank + self.qk_rope_head_dim
        if self.indexer is not None:
            values += self.indexer.values_per_token()

        return num_layers * values * bytes_per_element

    def dimension_rows(self) -> list[Row]:
        return [
            Row("Q/K head dimension", f"{self.qk_head_dim:,}"),
            Row("V head dimension", f"{self.v_head_dim:,}"),
        ]

    def summary_rows(self) -> list[Row]:
        rows = [
            Row("Type", self.attention_type),
            Row("Query heads", f"{self.num_attention_heads:,}"),
            Row("KV latent rank", f"{self.kv_lora_rank:,}"),
            Row("RoPE key dimension", f"{self.qk_rope_head_dim:,}"),
        ]

        if self.indexer is not None:
            rows.extend(
                [
                    Row("Indexer heads", f"{self.indexer.num_heads:,}"),
                    Row("Indexer key dim", f"{self.indexer.head_dim:,}"),
                    Row("Selected tokens", f"up to {self.indexer.top_k:,} per query"),
                ]
            )

        return rows

    def cache_layout_rows(self) -> list[Row]:
        layout = "Compressed MLA"
        if self.indexer is not None:
            layout = "Compressed MLA + indexer"

        return [Row("Layout", layout)]

    def derivation_sections(self, hidden_size: int) -> list[Section]:
        return [
            Section(
                "HEAD DIMENSIONS",
                [
                    (
                        f"Q/K: {self.qk_nope_head_dim:,} non-RoPE + "
                        f"{self.qk_rope_head_dim:,} RoPE"
                    ),
                    (
                        f"→ {self.qk_head_dim:,} per query/key head; "
                        f"{self.v_head_dim:,} per value head"
                    ),
                ],
            ),
            Section("ATTENTION", self._attention_lines()),
        ]

    def cache_derivation_lines(self, num_layers: int, dtype: DType) -> list[str]:
        terms = [
            f"{self.kv_lora_rank:,} KV latent",
            f"{self.qk_rope_head_dim:,} RoPE key",
        ]
        if self.indexer is not None:
            terms.append(f"{self.indexer.head_dim:,} indexer key")

        return [
            (
                f"{num_layers:,} layers × ({' + '.join(terms)}) × "
                f"{dtype.bytes_per_element} bytes ({dtype.label})"
            ),
        ]

    def assumptions(self) -> list[str]:
        if self.indexer is None:
            return [
                "MLA cache assumes compressed latents at the selected KV dtype.",
                "Backends storing expanded K/V use different memory.",
                "Quantization scales and metadata are excluded.",
            ]

        return [
            (
                "MLA cache assumes compressed latents and indexer keys at the "
                "selected KV dtype."
            ),
            "Backends storing expanded K/V or quantized index keys use different memory.",
            "Quantization scales and metadata are excluded.",
        ]

    def _attention_lines(self) -> list[str]:
        if self.indexer is None:
            return ["MLA shares a compressed KV latent across heads."]

        return [
            "DSA selects past tokens; MLA shares a compressed KV latent across heads.",
            (
                f"The indexer uses {self.indexer.num_heads:,} query heads and "
                "one shared key per token."
            ),
            (
                f"Top-k selects up to {self.indexer.top_k:,} tokens per query; "
                "it does not cap cache length."
            ),
        ]
