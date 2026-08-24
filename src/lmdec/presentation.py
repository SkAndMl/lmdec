from math import gcd

from lmdec.model_families import ModelAnalysis, ModelSpec


def render_analysis(analysis: ModelAnalysis) -> str:

    spec = analysis.spec

    fp32_size = _format_bytes(analysis.total_params * 4)
    bf16_size = _format_bytes(analysis.total_params * 2)
    fp16_size = _format_bytes(analysis.total_params * 2)

    kv_per_token_size = _format_bytes(analysis.kv_bytes_per_token)
    kv_at_8k_size = _format_bytes(analysis.kv_bytes_per_token * 8_192)
    kv_at_context_size = _format_bytes(
        analysis.kv_bytes_per_token * spec.context_window
    )

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
        _row("Weights (FP32)", f"~{fp32_size}"),
        _row("Weights (BF16)", f"~{bf16_size}"),
        _row("Weights (FP16)", f"~{fp16_size}"),
        "",
        "KV CACHE",
        _row("Per token", kv_per_token_size),
        _row("8K context", kv_at_8k_size),
    ]

    if spec.context_window != 8_192:
        lines.append(
            _row(
                _format_context_label(spec.context_window),
                kv_at_context_size,
            )
        )

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


def _format_context_label(context_window: int) -> str:
    if context_window % 1_024 == 0:
        return f"{context_window // 1_024}K context"

    return f"{context_window:,} context"


def _head_ratio(spec: ModelSpec) -> str:
    divisor = gcd(spec.num_attention_heads, spec.num_kv_heads)
    query_ratio = spec.num_attention_heads // divisor
    kv_ratio = spec.num_kv_heads // divisor
    return f"{query_ratio}:{kv_ratio}"
