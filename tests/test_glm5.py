from dataclasses import replace

import pytest
import torch
from transformers import AutoModelForCausalLM, GlmMoeDsaConfig

from lmdec.model_families.base import DType
from lmdec.model_families.glm5 import GLM5Family
from lmdec.model_families.registry import resolve_family
from lmdec.presentation import render_analysis


@pytest.fixture
def config() -> GlmMoeDsaConfig:
    # Published zai-org/GLM-5 dimensions, kept local so tests need no downloads.
    return GlmMoeDsaConfig(
        architectures=["GlmMoeDsaForCausalLM"],
        vocab_size=154_880,
        hidden_size=6_144,
        num_hidden_layers=78,
        num_attention_heads=64,
        num_key_value_heads=64,
        max_position_embeddings=202_752,
        intermediate_size=12_288,
        first_k_dense_replace=3,
        q_lora_rank=2_048,
        kv_lora_rank=512,
        qk_nope_head_dim=192,
        qk_rope_head_dim=64,
        v_head_dim=256,
        index_head_dim=128,
        index_n_heads=32,
        index_topk=2_048,
        n_routed_experts=256,
        n_shared_experts=1,
        num_experts_per_tok=8,
        moe_intermediate_size=2_048,
        tie_word_embeddings=False,
    )


@pytest.mark.parametrize("tied", [False, True])
def test_glm5_parameter_estimate_matches_model_shapes(
    config: GlmMoeDsaConfig, tied: bool
) -> None:
    config.tie_word_embeddings = tied
    family = GLM5Family("zai-org/GLM-5", config)
    # Meta tensors verify real model shapes without allocating the weights.
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config)
    expected = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if "norm" not in name and "bias" not in name
    )

    assert family.estimate_params() == expected
    if not tied:
        assert expected == 743_910_014_976


@pytest.mark.parametrize(
    ("kv_dtype", "expected_bytes", "per_token"),
    [
        ("bf16", 109_824, "107.25 KiB"),
        ("fp16", 109_824, "107.25 KiB"),
        ("fp32", 219_648, "214.5 KiB"),
    ],
)
@pytest.mark.parametrize("explain", [False, True])
def test_glm5_registry_analysis_and_rendering(
    config: GlmMoeDsaConfig,
    kv_dtype: DType,
    expected_bytes: int,
    per_token: str,
    explain: bool,
) -> None:
    family = resolve_family("zai-org/GLM-5", config)
    assert isinstance(family, GLM5Family)

    analysis = family.analyze(
        context=4_096, batch_size=2, dtype="bf16", kv_dtype=kv_dtype, explain=explain
    )
    report = render_analysis(analysis)

    assert analysis.spec is family.spec
    assert analysis.total_params == 743_910_014_976
    assert analysis.kv_bytes_per_token == expected_bytes
    assert "~743.91B" in report
    assert "Q/K head dimension  256" in report
    assert "V head dimension    256" in report
    assert "DSA (MLA)" in report
    assert "Compressed MLA + indexer" in report
    assert per_token in report
    assert "Q:KV ratio" not in report
    assert ("DERIVATION" in report) is explain
    if kv_dtype != "fp32":
        # 4096-token context exceeds top-k: storage must still cover all tokens.
        assert "858 MiB" in report
    if explain:
        element_bytes = 4 if kv_dtype == "fp32" else 2
        assert (
            "78 layers × (512 KV latent + 64 RoPE key + 128 indexer key) × "
            f"{element_bytes} bytes ({kv_dtype.upper()})"
        ) in report
        assert "192 non-RoPE + 64 RoPE" in report
        assert "one shared key per token" in report
        assert "does not cap cache length" in report
        assert "expanded K/V" in report
        assert "Quantization scales and metadata are excluded." in report
        assert "2 (K + V)" not in report
        assert "6,144 / 64" not in report
        assert "no KV-head sharing" not in report


def test_glm5_compressed_cache_does_not_store_queries_or_expanded_heads(
    config: GlmMoeDsaConfig,
) -> None:
    family = GLM5Family("zai-org/GLM-5", config)
    family.spec = replace(
        family.spec,
        num_attention_heads=32,
        num_kv_heads=32,
        index_n_heads=16,
        index_topk=1_024,
        q_lora_rank=1_024,
        v_head_dim=128,
    )

    assert family.calculate_kv_cache_bytes(bytes_per_value=2) == 109_824


def test_glm5_cache_scales_with_layers_and_stored_dimensions(
    config: GlmMoeDsaConfig,
) -> None:
    family = GLM5Family("zai-org/GLM-5", config)
    # Two layers, each storing 16 latent + 8 RoPE + 8 indexer values.
    family.spec = replace(
        family.spec, num_layers=2, kv_lora_rank=16, qk_rope_head_dim=8, index_head_dim=8
    )

    assert family.calculate_kv_cache_bytes(bytes_per_value=4) == 256
