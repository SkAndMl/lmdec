def attention_type(num_kv_heads: int, num_attention_heads: int) -> str:
    if num_kv_heads == num_attention_heads:
        return "MHA"

    if num_kv_heads == 1:
        return "MQA"

    return "GQA"


def dtype_to_byte_count(dtype) -> int:
    match dtype:
        case "bf16" | "fp16":
            return 2
        case "fp32":
            return 4
        case _:
            raise ValueError(f"Unsupported dtype: {dtype}")
