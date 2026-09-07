from transformers import Qwen2Config

from lmdec.model_families.base import Workload
from lmdec.model_families.qwen2 import Qwen2Family
from lmdec.presentation import render_analysis


def _family() -> Qwen2Family:
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
    return Qwen2Family("Qwen/Qwen2.5-0.5B", config)


def test_render_analysis_matches_expected_qwen_report() -> None:
    analysis = _family().analyze(
        Workload(context=4_096, batch_size=3, dtype="fp32", kv_dtype="bf16")
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
KV cache (BF16)     144 MiB
Total               ~1.98 GiB

KV CACHE (BF16)
Per token           12 KiB
Context             4,096 tokens
Batch size          3
Total               144 MiB

ATTENTION
Type                GQA
Query heads         14
KV heads            2
Q:KV ratio          7:1"""


def test_render_analysis_appends_derivations_when_explain_is_enabled() -> None:
    analysis = _family().analyze(
        Workload(context=4_096, batch_size=3, dtype="fp32", kv_dtype="bf16")
    )

    report = render_analysis(analysis, explain=True)

    assert report.endswith(
        """DERIVATION
────────────────────────────────────────

PARAMETERS
Qwen2ForCausalLM architecture estimate
→ 493,961,216 parameters
Input and output embeddings are tied; the LM head adds no
separate 151,936 × 896 parameter matrix.

HEAD DIMENSION
896 / 14 query heads
→ 64

ATTENTION
14 query heads / 2 KV heads
→ GQA, Q:KV ratio 7:1
→ Each KV head is shared by 7 query heads, reducing KV-cache
  payload by 7× versus equivalent MHA.

WEIGHT MEMORY
493,961,216 parameters × 4 bytes (FP32)
→ 1.84 GiB

KV CACHE
2 (K + V) × 24 layers × 2 KV heads × 64 × 2 bytes (BF16)
→ 12 KiB/token

12 KiB/token × 4,096 tokens
→ 48 MiB/sequence

12 KiB/token × 4,096 tokens × 3 sequences
→ 144 MiB total KV cache

ASSUMPTIONS
────────────────────────────────────────
Weight memory is raw parameter storage only.
KV memory is theoretical tensor payload.
Batch size 3 means 3 sequences, each occupying the full
4,096-token context.
Activations, temporary workspaces, allocator/runtime overhead,
cache block rounding, and runtime-specific layouts are excluded."""
    )


def test_render_analysis_omits_derivations_by_default() -> None:
    analysis = _family().analyze(Workload(context=4_096, batch_size=3))

    assert "DERIVATION" not in render_analysis(analysis)


def test_render_analysis_explains_single_sequence_batch_without_repeating_total(
) -> None:
    analysis = _family().analyze(Workload(context=32_768, batch_size=1))

    report = render_analysis(analysis, explain=True)

    assert "→ 384 MiB/sequence" in report
    assert "Batch size 1 means one sequence occupying the full" in report
    assert "× 1 sequence" not in report
