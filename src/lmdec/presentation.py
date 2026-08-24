from math import gcd

from lmdec.model_families import ModelAnalysis, ModelSpec
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
    dtype_label = analysis.dtype.upper()

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
        _row("KV heads", f"{spec.num_kv_heads:,}"),
        _row("Head dimension", f"{spec.head_dim:,}"),
        _row("Max context", f"{spec.context_window:,}"),
        "",
        "MEMORY",
        _row(f"Weights ({dtype_label})", f"~{weight_size}"),
        _row("KV cache", kv_cache_size),
        _row("Total", f"~{total_size}"),
        "",
        f"KV CACHE ({dtype_label})",
        _row("Per token", kv_per_token_size),
        _row("Context", f"{analysis.context:,} tokens"),
        _row("Batch size", f"{analysis.batch_size:,}"),
        _row("Total", kv_cache_size),
    ]

    lines.extend(
        [
            "",
            "ATTENTION",
            _row("Type", spec.attention_type),
            _row("Query heads", f"{spec.num_attention_heads:,}"),
            _row("KV heads", f"{spec.num_kv_heads:,}"),
            _row("Q:KV ratio", _head_ratio(spec)),
        ]
    )

    return "\n".join(lines)


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
