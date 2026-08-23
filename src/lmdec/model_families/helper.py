def attention_type(num_kv_heads: int, num_attention_heads: int) -> str:
    if num_kv_heads == num_attention_heads:
        return "MHA"

    if num_kv_heads == 1:
        return "MQA"

    return "GQA"
