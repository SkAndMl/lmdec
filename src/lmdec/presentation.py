from math import gcd

from lmdec.model_families.base import (
    BaseModelSpec,
    GLM5Spec,
    ModelAnalysis,
    ModelSpec,
)
from lmdec.model_families.helper import dtype_to_byte_count


def render_analysis(analysis: ModelAnalysis) -> str:

    spec = analysis.spec

    weight_bytes = analysis.total_params * dtype_to_byte_count(analysis.dtype)

    kv_per_token_size = _format_bytes(analysis.kv_bytes_per_token)
    kv_cache_bytes = (
        analysis.kv_bytes_per_token * analysis.context * analysis.batch_size
    )

    weight_size = _format_bytes(weight_bytes)
    kv_cache_size = _format_bytes(kv_cache_bytes)
    total_size = _format_bytes(weight_bytes + kv_cache_bytes)
    weight_dtype_label = analysis.dtype.upper()
    kv_dtype_label = analysis.kv_dtype.upper()

    lines = [
        spec.model_id,
        "─" * 40,
        "",
        "MODEL",
        _row("Architecture", spec.architecture),
        _row("Parameters", f"~{_format_params(analysis.total_params)}"),
        _row("Layers", f"{spec.num_layers:,}"),
        _row("Hidden size", f"{spec.hidden_size:,}"),
        _row("Attention heads", f"{spec.num_attention_heads:,}"),
        *_model_dimensions(spec),
        _row("Max context", f"{spec.context_window:,}"),
        "",
        "MEMORY",
        _row(f"Weights ({weight_dtype_label})", f"~{weight_size}"),
        _row(f"KV cache ({kv_dtype_label})", kv_cache_size),
        _row("Total", f"~{total_size}"),
        "",
        f"KV CACHE ({kv_dtype_label})",
        *(
            [_row("Layout", "Compressed MLA + indexer")]
            if isinstance(spec, GLM5Spec)
            else []
        ),
        _row("Per token", kv_per_token_size),
        _row("Context", f"{analysis.context:,} tokens"),
        _row("Batch size", f"{analysis.batch_size:,}"),
        _row("Total", kv_cache_size),
    ]

    lines.extend(_attention_rows(spec))

    if analysis.explain:
        lines.extend(["", *_render_derivation(analysis)])

    return "\n".join(lines)


def _render_derivation(analysis: ModelAnalysis) -> list[str]:
    spec = analysis.spec
    weight_bytes_per_parameter = dtype_to_byte_count(analysis.dtype)
    kv_bytes_per_element = dtype_to_byte_count(analysis.kv_dtype)
    kv_bytes_per_sequence = analysis.kv_bytes_per_token * analysis.context
    kv_cache_bytes = kv_bytes_per_sequence * analysis.batch_size

    lines = [
        "DERIVATION",
        "─" * 40,
        "",
        "PARAMETERS",
        f"{spec.architecture} architecture estimate",
        f"→ {analysis.total_params:,} parameters",
        *_embedding_assumption(spec),
        *_attention_derivation(spec),
        "",
        "WEIGHT MEMORY",
        (
            f"{analysis.total_params:,} parameters × "
            f"{weight_bytes_per_parameter} bytes ({analysis.dtype.upper()})"
        ),
        f"→ {_format_bytes(analysis.total_params * weight_bytes_per_parameter)}",
        "",
        "KV CACHE",
        *_cache_derivation(spec, kv_bytes_per_element, analysis.kv_dtype.upper()),
        f"→ {_format_bytes(analysis.kv_bytes_per_token)}/token",
        "",
        (
            f"{_format_bytes(analysis.kv_bytes_per_token)}/token × "
            f"{analysis.context:,} tokens"
        ),
        f"→ {_format_bytes(kv_bytes_per_sequence)}/sequence",
    ]

    if analysis.batch_size > 1:
        lines.extend(
            [
                "",
                (
                    f"{_format_bytes(analysis.kv_bytes_per_token)}/token × "
                    f"{analysis.context:,} tokens × "
                    f"{analysis.batch_size:,} sequences"
                ),
                f"→ {_format_bytes(kv_cache_bytes)} total KV cache",
            ]
        )

    lines.extend(
        [
            "",
            "ASSUMPTIONS",
            "─" * 40,
            "Weight memory is raw parameter storage only.",
            "KV memory is theoretical tensor payload.",
            *(
                [
                    "GLM cache assumes compressed MLA and indexer keys at the selected KV dtype.",
                    "Backends storing expanded K/V or quantized index keys use different memory.",
                    "Quantization scales and metadata are excluded.",
                ]
                if isinstance(spec, GLM5Spec)
                else []
            ),
            *_batch_assumption(analysis),
            "Activations, temporary workspaces, allocator/runtime overhead,",
            "cache block rounding, and runtime-specific layouts are excluded.",
        ]
    )

    return lines


def _model_dimensions(spec: BaseModelSpec) -> list[str]:
    if isinstance(spec, GLM5Spec):
        return [
            _row("Q/K head dimension", f"{spec.qk_head_dim:,}"),
            _row("V head dimension", f"{spec.v_head_dim:,}"),
        ]
    if isinstance(spec, ModelSpec):
        return [
            _row("KV heads", f"{spec.num_kv_heads:,}"),
            _row("Head dimension", f"{spec.head_dim:,}"),
        ]
    return []


def _attention_rows(spec: BaseModelSpec) -> list[str]:
    if isinstance(spec, GLM5Spec):
        return [
            "",
            "ATTENTION",
            _row("Type", "DSA (MLA)"),
            _row("Query heads", f"{spec.num_attention_heads:,}"),
            _row("KV latent rank", f"{spec.kv_lora_rank:,}"),
            _row("RoPE key dimension", f"{spec.qk_rope_head_dim:,}"),
            _row("Indexer heads", f"{spec.index_n_heads:,}"),
            _row("Indexer key dim", f"{spec.index_head_dim:,}"),
            _row("Selected tokens", f"up to {spec.index_topk:,} per query"),
        ]
    if isinstance(spec, ModelSpec):
        return [
            "",
            "ATTENTION",
            _row("Type", spec.attention_type),
            _row("Query heads", f"{spec.num_attention_heads:,}"),
            _row("KV heads", f"{spec.num_kv_heads:,}"),
            _row("Q:KV ratio", _head_ratio(spec)),
        ]
    return []


def _attention_derivation(spec: BaseModelSpec) -> list[str]:
    if isinstance(spec, GLM5Spec):
        return [
            "",
            "HEAD DIMENSIONS",
            f"Q/K: {spec.qk_nope_head_dim:,} non-RoPE + {spec.qk_rope_head_dim:,} RoPE",
            f"→ {spec.qk_head_dim:,} per query/key head; {spec.v_head_dim:,} per value head",
            "",
            "ATTENTION",
            "DSA selects past tokens; MLA shares a compressed KV latent across heads.",
            f"The indexer uses {spec.index_n_heads:,} query heads and one shared key per token.",
            f"Top-k selects up to {spec.index_topk:,} tokens per query; it does not cap cache length.",
        ]
    if isinstance(spec, ModelSpec):
        return [
            "",
            "HEAD DIMENSION",
            f"{spec.hidden_size:,} / {spec.num_attention_heads:,} query heads",
            f"→ {spec.head_dim:,}",
            "",
            "ATTENTION",
            f"{spec.num_attention_heads:,} query heads / {spec.num_kv_heads:,} KV heads",
            f"→ {spec.attention_type}, Q:KV ratio {_head_ratio(spec)}",
            *_attention_implication(spec),
        ]
    return []


def _cache_derivation(
    spec: BaseModelSpec, bytes_per_element: int, dtype_label: str
) -> list[str]:
    if isinstance(spec, GLM5Spec):
        return [
            (
                f"{spec.num_layers:,} layers × ({spec.kv_lora_rank:,} KV latent + "
                f"{spec.qk_rope_head_dim:,} RoPE key + {spec.index_head_dim:,} indexer key) × "
                f"{bytes_per_element} bytes ({dtype_label})"
            ),
        ]
    if isinstance(spec, ModelSpec):
        return [
            (
                f"2 (K + V) × {spec.num_layers:,} layers × "
                f"{spec.num_kv_heads:,} KV heads × {spec.head_dim:,} × "
                f"{bytes_per_element} bytes ({dtype_label})"
            ),
        ]
    return []


def _embedding_assumption(spec: BaseModelSpec) -> list[str]:
    matrix_size = f"{spec.vocab_size:,} × {spec.hidden_size:,}"
    if spec.tie_word_embeddings:
        return [
            "Input and output embeddings are tied; the LM head adds no",
            f"separate {matrix_size} parameter matrix.",
        ]

    return [
        "Input and output embeddings are untied; the estimate includes a",
        f"separate {matrix_size} parameter matrix for the LM head.",
    ]


def _attention_implication(spec: ModelSpec) -> list[str]:
    query_heads = spec.num_attention_heads
    kv_heads = spec.num_kv_heads

    if query_heads == kv_heads:
        return ["→ Each query head has its own KV head; no KV-head sharing."]

    if query_heads % kv_heads == 0:
        sharing_factor = query_heads // kv_heads
        return [
            (
                f"→ Each KV head is shared by {sharing_factor:,} query heads, "
                "reducing KV-cache"
            ),
            f"  payload by {sharing_factor:,}× versus equivalent MHA.",
        ]

    return ["→ Fewer KV heads than query heads reduce KV-cache payload versus MHA."]


def _batch_assumption(analysis: ModelAnalysis) -> list[str]:
    if analysis.batch_size == 1:
        return [
            "Batch size 1 means one sequence occupying the full",
            f"{analysis.context:,}-token context.",
        ]

    return [
        (
            f"Batch size {analysis.batch_size:,} means "
            f"{analysis.batch_size:,} sequences, each occupying the full"
        ),
        f"{analysis.context:,}-token context.",
    ]


def _row(label: str, value: str) -> str:
    return f"{label:<20}{value}"


def _format_params(total_params: int) -> str:
    size, unit = _scale(total_params, base=1_000, units=("", "K", "M", "B", "T"))
    return f"{_format_number(size)}{unit}"


def _format_bytes(num_bytes: int) -> str:
    size, unit = _scale(
        num_bytes,
        base=1_024,
        units=("B", "KiB", "MiB", "GiB", "TiB", "PiB"),
    )
    return f"{_format_number(size)} {unit}"


def _scale(value: int, base: int, units: tuple[str, ...]) -> tuple[float, str]:
    if value < 0:
        raise ValueError(f"value ({value}) cannot be less than 0")

    size = float(value)
    unit_index = 0

    while size >= base and unit_index < len(units) - 1:
        size /= base
        unit_index += 1

    if round(size, 2) >= base and unit_index < len(units) - 1:
        size /= base
        unit_index += 1

    return size, units[unit_index]


def _format_number(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _head_ratio(spec: ModelSpec) -> str:
    divisor = gcd(spec.num_attention_heads, spec.num_kv_heads)
    query_ratio = spec.num_attention_heads // divisor
    kv_ratio = spec.num_kv_heads // divisor
    return f"{query_ratio}:{kv_ratio}"
