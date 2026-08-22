from lmdec.analyze import calculate_kv_bytes, estimate_params
from lmdec.model_families import ModelSpec
from lmdec.presentation import render_analysis


def test_render_analysis_matches_expected_qwen_report() -> None:
    model_spec = ModelSpec(
        model_id="Qwen/Qwen2.5-0.5B",
        architecture="Qwen2ForCausalLM",
        model_type="qwen2",
        hidden_size=896,
        intermediate_size=4_864,
        num_layers=24,
        num_attention_heads=14,
        num_kv_heads=2,
        head_dim=64,
        vocab_size=151_936,
        context_window=32_768,
        tie_word_embeddings=True,
        gated_mlp=True,
    )
    total_params = estimate_params(model_spec)
    kv_bytes_per_token = calculate_kv_bytes(
        num_kv_heads=model_spec.num_kv_heads,
        head_dim=model_spec.head_dim,
        bytes_per_value=2,
        num_layers=model_spec.num_layers,
    )

    report = render_analysis(model_spec, total_params, kv_bytes_per_token)

    assert report == """Qwen/Qwen2.5-0.5B
────────────────────────────────────────

MODEL
Architecture        Qwen2ForCausalLM
Parameters          ~494M
Layers              24
Hidden size         896
Attention heads     14
KV heads            2
Head dimension      64
Max context         32,768

MEMORY
Weights (FP32)      ~1.84 GiB
Weights (BF16)      ~0.92 GiB
Weights (FP16)      ~0.92 GiB

KV CACHE
Per token           12.0 KiB
8K context          96 MiB
32K context         384 MiB

ATTENTION
Type                GQA
Query heads         14
KV heads            2
Q:KV ratio          7:1"""
