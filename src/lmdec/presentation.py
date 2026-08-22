from math import gcd

from lmdec.model_families import ModelSpec


def render_analysis(
    model_spec: ModelSpec,
    total_params: int,
    kv_bytes_per_token: int,
) -> str:

    fp32_size = _format_gib(total_params * 4)
    bf16_size = _format_gib(total_params * 2)
    fp16_size = _format_gib(total_params * 2)

    kv_per_token_size = _format_bytes(kv_bytes_per_token, decimal_places=1)
    kv_at_8k_size = _format_bytes(kv_bytes_per_token * 8_192)
    kv_at_context_size = _format_bytes(kv_bytes_per_token * model_spec.context_window)

    lines = [
        model_spec.model_id,
        "─" * 40,
        "",
        "MODEL",
        _row("Architecture", model_spec.architecture),
        _row("Parameters", _format_params(total_params)),
        _row("Layers", f"{model_spec.num_layers:,}"),
        _row("Hidden size", f"{model_spec.hidden_size:,}"),
        _row("Attention heads", f"{model_spec.num_attention_heads:,}"),
        _row("KV heads", f"{model_spec.num_kv_heads:,}"),
        _row("Head dimension", f"{model_spec.head_dim:,}"),
        _row("Max context", f"{model_spec.context_window:,}"),
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

    if model_spec.context_window != 8_192:
        lines.append(
            _row(
                _format_context_label(model_spec.context_window),
                kv_at_context_size,
            )
        )

    lines.extend(
        [
            "",
            "ATTENTION",
            _row("Type", _attention_type(model_spec)),
            _row("Query heads", f"{model_spec.num_attention_heads:,}"),
            _row("KV heads", f"{model_spec.num_kv_heads:,}"),
            _row("Q:KV ratio", _head_ratio(model_spec)),
        ]
    )

    return "\n".join(lines)


def _row(label: str, value: str) -> str:
    return f"{label:<20}{value}"


def _format_params(total_params: int) -> str:
    if total_params >= 1_000_000_000:
        return f"~{_format_number(total_params / 1_000_000_000)}B"

    if total_params >= 1_000_000:
        return f"~{total_params / 1_000_000:.0f}M"

    if total_params >= 1_000:
        return f"~{total_params / 1_000:.0f}K"

    return f"~{total_params}"


def _get_resized_num_bytes_and_units(num_bytes: int) -> tuple[float, str]:

    if num_bytes < 0:
        raise ValueError(f"num_bytes ({num_bytes}) cannot be less than 0")

    match num_bytes:
        case x if x < 1024:
            return x, "B"
        case x if x < 1024 * 1024:
            return round(x / 1024, 2), "KiB"
        case x if x < 1024 * 1024 * 1024:
            return round(x / (1024 * 1024), 2), "MiB"

    return round(x / (1024 * 1024 * 1024), 2), "GiB"


def _format_bytes(num_bytes: int, decimal_places: int | None = None) -> str:
    size, unit = _get_resized_num_bytes_and_units(num_bytes)

    if decimal_places is not None:
        size_text = f"{size:.{decimal_places}f}"
    else:
        size_text = _format_number(size)

    return f"{size_text} {unit}"


def _format_gib(num_bytes: int) -> str:
    size = num_bytes / (1_024**3)
    return f"{size:.2f} GiB"


def _format_number(value: float) -> str:
    if value.is_integer():
        return str(int(value))

    return str(value)


def _format_context_label(context_window: int) -> str:
    if context_window % 1_024 == 0:
        return f"{context_window // 1_024}K context"

    return f"{context_window:,} context"


def _attention_type(model_spec: ModelSpec) -> str:
    if model_spec.num_kv_heads == model_spec.num_attention_heads:
        return "MHA"

    if model_spec.num_kv_heads == 1:
        return "MQA"

    return "GQA"


def _head_ratio(model_spec: ModelSpec) -> str:
    divisor = gcd(model_spec.num_attention_heads, model_spec.num_kv_heads)
    query_ratio = model_spec.num_attention_heads // divisor
    kv_ratio = model_spec.num_kv_heads // divisor
    return f"{query_ratio}:{kv_ratio}"
