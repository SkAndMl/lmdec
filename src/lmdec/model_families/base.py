from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from lmdec.report import Row, Section


class DType(StrEnum):
    BF16 = "bf16"
    FP16 = "fp16"
    FP32 = "fp32"

    @property
    def bytes_per_element(self) -> int:
        return _BYTES_PER_ELEMENT[self]

    @property
    def label(self) -> str:
        return self.value.upper()


_BYTES_PER_ELEMENT: dict[DType, int] = {
    DType.BF16: 2,
    DType.FP16: 2,
    DType.FP32: 4,
}


class AttentionSpec(Protocol):
    num_attention_heads: int

    def parameters(self, hidden_size: int, num_layers: int) -> int: ...

    def kv_bytes_per_token(self, num_layers: int, bytes_per_element: int) -> int: ...

    def dimension_rows(self) -> list[Row]: ...

    def summary_rows(self) -> list[Row]: ...

    def cache_layout_rows(self) -> list[Row]: ...

    def derivation_sections(self, hidden_size: int) -> list[Section]: ...

    def cache_derivation_lines(self, num_layers: int, dtype: DType) -> list[str]: ...

    def assumptions(self) -> list[str]: ...


class MLPSpec(Protocol):
    def parameters(self, hidden_size: int, num_layers: int) -> int: ...


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    architecture: str
    model_type: str

    hidden_size: int
    num_layers: int
    vocab_size: int
    context_window: int
    tie_word_embeddings: bool

    attention: AttentionSpec
    mlp: MLPSpec

    def __post_init__(self) -> None:
        if self.hidden_size <= 0:
            raise ValueError(f"hidden_size ({self.hidden_size}) must be positive")
        if self.num_layers <= 0:
            raise ValueError(f"num_layers ({self.num_layers}) must be positive")
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size ({self.vocab_size}) must be positive")

    @property
    def num_attention_heads(self) -> int:
        return self.attention.num_attention_heads

    def parameter_breakdown(self) -> "ParameterBreakdown":
        embedding = self.vocab_size * self.hidden_size
        return ParameterBreakdown(
            embedding=embedding,
            attention=self.attention.parameters(self.hidden_size, self.num_layers),
            mlp=self.mlp.parameters(self.hidden_size, self.num_layers),
            lm_head=0 if self.tie_word_embeddings else embedding,
        )

    def kv_bytes_per_token(self, kv_dtype: DType) -> int:
        return self.attention.kv_bytes_per_token(
            self.num_layers, kv_dtype.bytes_per_element
        )


@dataclass(frozen=True)
class ParameterBreakdown:
    embedding: int
    attention: int
    mlp: int
    lm_head: int

    @property
    def total(self) -> int:
        return self.embedding + self.attention + self.mlp + self.lm_head


@dataclass(frozen=True)
class MemoryBreakdown:
    weight_bytes: int
    kv_bytes_per_token: int
    kv_bytes_per_sequence: int
    kv_bytes_total: int

    @property
    def total_bytes(self) -> int:
        return self.weight_bytes + self.kv_bytes_total


@dataclass(frozen=True)
class Workload:
    context: int = 1
    batch_size: int = 1
    dtype: DType = DType.BF16
    kv_dtype: DType = DType.BF16

    def __post_init__(self) -> None:
        if self.context <= 0:
            raise ValueError(f"context ({self.context}) must be positive")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size ({self.batch_size}) must be positive")

        object.__setattr__(self, "dtype", DType(self.dtype))
        object.__setattr__(self, "kv_dtype", DType(self.kv_dtype))


@dataclass(frozen=True)
class ModelAnalysis:
    spec: ModelSpec
    workload: Workload
    params: ParameterBreakdown
    memory: MemoryBreakdown


def analyze(spec: ModelSpec, workload: Workload) -> ModelAnalysis:
    params = spec.parameter_breakdown()
    kv_bytes_per_token = spec.kv_bytes_per_token(workload.kv_dtype)
    kv_bytes_per_sequence = kv_bytes_per_token * workload.context

    return ModelAnalysis(
        spec=spec,
        workload=workload,
        params=params,
        memory=MemoryBreakdown(
            weight_bytes=params.total * workload.dtype.bytes_per_element,
            kv_bytes_per_token=kv_bytes_per_token,
            kv_bytes_per_sequence=kv_bytes_per_sequence,
            kv_bytes_total=kv_bytes_per_sequence * workload.batch_size,
        ),
    )


class ModelFamily(Protocol):
    spec: ModelSpec

    def analyze(self, workload: Workload) -> ModelAnalysis: ...


class BaseFamily:
    spec: ModelSpec

    def analyze(self, workload: Workload) -> ModelAnalysis:
        return analyze(self.spec, workload)
