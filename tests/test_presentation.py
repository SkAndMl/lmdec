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
    analysis = family.analyze(
        context=4_096,
        batch_size=3,
        dtype="fp32",
    )

    report = render_analysis(analysis)

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
KV cache            288 MiB
Total               ~2.12 GiB

KV CACHE (FP32)
Per token           24 KiB
Context             4,096 tokens
Batch size          3
Total               288 MiB

ATTENTION
Type                GQA
Query heads         14
KV heads            2
Q:KV ratio          7:1"""
