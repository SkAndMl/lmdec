from transformers import Qwen2Config

from lmdec.model_families.qwen2 import Qwen2Family
from lmdec.presentation import render_analysis


def test_render_analysis_matches_expected_qwen_report() -> None:
    config = Qwen2Config(
        architectures=["Qwen2ForCausalLM"],
        vocab_size=151_936,
        hidden_size=896,
        intermediate_size=4_864,
        num_hidden_layers=24,
        num_attention_heads=14,
        num_key_value_heads=2,
        max_position_embeddings=32_768,
        tie_word_embeddings=True,
    )
    family = Qwen2Family("Qwen/Qwen2.5-0.5B", config)
    total_params = family.estimate_params()
    kv_bytes_per_token = family.calculate_kv_cache_bytes(
        context_length=1,
        bytes_per_token=2,
    )

    report = render_analysis(family, total_params, kv_bytes_per_token)

    assert report == """Qwen/Qwen2.5-0.5B
────────────────────────────────────────

MODEL
Architecture        Qwen2ForCausalLM
Parameters          ~493.96M
Layers              24
Hidden size         896
Attention heads     14
KV heads            2
Head dimension      64
Max context         32,768

MEMORY
Weights (FP32)      ~1.84 GiB
Weights (BF16)      ~942.16 MiB
Weights (FP16)      ~942.16 MiB

KV CACHE
Per token           12 KiB
8K context          96 MiB
32K context         384 MiB

ATTENTION
Type                GQA
Query heads         14
KV heads            2
Q:KV ratio          7:1"""
