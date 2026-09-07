import pytest
import torch
from transformers import AutoModelForCausalLM, DeepseekV3Config, DeepseekV32Config

from lmdec.model_families.base import DType, Workload
from lmdec.model_families.deepseek_v3 import DeepseekV3Family, DeepseekV32Family
from lmdec.model_families.registry import resolve_family
from lmdec.presentation import render_analysis

# Published deepseek-ai/DeepSeek-V3 dimensions, kept local so tests need no downloads.
COMMON = {
    "vocab_size": 129_280,
    "hidden_size": 7_168,
    "intermediate_size": 18_432,
    "moe_intermediate_size": 2_048,
    "num_hidden_layers": 61,
    "num_attention_heads": 128,
    "n_shared_experts": 1,
    "n_routed_experts": 256,
    "num_experts_per_tok": 8,
    "first_k_dense_replace": 3,
    "q_lora_rank": 1_536,
    "kv_lora_rank": 512,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "v_head_dim": 128,
    "max_position_embeddings": 163_840,
    "tie_word_embeddings": False,
}

V3_PARAMS = 671_025_397_760
V32_PARAMS = 671_876_907_008

# 61 layers × (512 latent + 64 RoPE key [+ 128 indexer key]) × 2 bytes.
V3_KV_BYTES = 70_272
V32_KV_BYTES = 85_888


@pytest.fixture
def v3_config() -> DeepseekV3Config:
    return DeepseekV3Config(architectures=["DeepseekV3ForCausalLM"], **COMMON)


@pytest.fixture
def v32_config() -> DeepseekV32Config:
    return DeepseekV32Config(
        architectures=["DeepseekV32ForCausalLM"],
        index_head_dim=128,
        index_n_heads=64,
        index_topk=2_048,
        **COMMON,
    )


@pytest.mark.parametrize("tied", [False, True])
def test_deepseek_v3_parameter_estimate_matches_model_shapes(
    v3_config: DeepseekV3Config, tied: bool
) -> None:
    v3_config.tie_word_embeddings = tied
    family = DeepseekV3Family("deepseek-ai/DeepSeek-V3", v3_config)
    # Meta tensors verify real model shapes without allocating the weights.
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(v3_config)
    expected = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if "norm" not in name and "bias" not in name
    )

    assert family.spec.parameter_breakdown().total == expected
    if not tied:
        assert expected == V3_PARAMS


@pytest.mark.parametrize("tied", [False, True])
def test_deepseek_v32_parameter_estimate_matches_model_shapes(
    v32_config: DeepseekV32Config, tied: bool
) -> None:
    v32_config.tie_word_embeddings = tied
    family = DeepseekV32Family("deepseek-ai/DeepSeek-V3.2-Exp", v32_config)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(v32_config)
    expected = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if "norm" not in name and "bias" not in name
    )

    assert family.spec.parameter_breakdown().total == expected
    if not tied:
        assert expected == V32_PARAMS


def test_deepseek_v32_adds_only_indexer_parameters_over_v3(
    v3_config: DeepseekV3Config, v32_config: DeepseekV32Config
) -> None:
    v3 = DeepseekV3Family("deepseek-ai/DeepSeek-V3", v3_config).spec
    v32 = DeepseekV32Family("deepseek-ai/DeepSeek-V3.2-Exp", v32_config).spec
    indexer = v32.attention.indexer

    assert v3.attention.indexer is None
    assert v3.mlp == v32.mlp
    assert v32.parameter_breakdown().attention - v3.parameter_breakdown().attention == (
        v3.num_layers * indexer.parameters(v3.hidden_size, v32.attention.q_lora_rank)
    )


@pytest.mark.parametrize(
    ("kv_dtype", "expected_v3", "expected_v32"),
    [
        (DType.BF16, V3_KV_BYTES, V32_KV_BYTES),
        (DType.FP16, V3_KV_BYTES, V32_KV_BYTES),
        (DType.FP32, 2 * V3_KV_BYTES, 2 * V32_KV_BYTES),
    ],
)
def test_deepseek_kv_cache_counts_indexer_key_only_for_v32(
    v3_config: DeepseekV3Config,
    v32_config: DeepseekV32Config,
    kv_dtype: DType,
    expected_v3: int,
    expected_v32: int,
) -> None:
    v3 = DeepseekV3Family("deepseek-ai/DeepSeek-V3", v3_config).spec
    v32 = DeepseekV32Family("deepseek-ai/DeepSeek-V3.2-Exp", v32_config).spec

    assert v3.kv_bytes_per_token(kv_dtype) == expected_v3
    assert v32.kv_bytes_per_token(kv_dtype) == expected_v32


def test_deepseek_v3_renders_plain_mla_without_indexer(
    v3_config: DeepseekV3Config,
) -> None:
    family = resolve_family("deepseek-ai/DeepSeek-V3", v3_config)
    assert isinstance(family, DeepseekV3Family)

    analysis = family.analyze(Workload(context=8_192, batch_size=1))
    report = render_analysis(analysis, explain=True)

    assert analysis.params.total == V3_PARAMS
    assert analysis.memory.kv_bytes_per_token == V3_KV_BYTES
    assert "~671.03B" in report
    assert "Type                MLA" in report
    assert "Layout              Compressed MLA" in report
    assert "Q/K head dimension  192" in report
    assert "V head dimension    128" in report
    assert "Per token           68.62 KiB" in report
    assert "61 layers × (512 KV latent + 64 RoPE key) × 2 bytes (BF16)" in report
    assert "MLA shares a compressed KV latent across heads." in report
    assert "MLA cache assumes compressed latents at the selected KV dtype." in report
    assert "DSA" not in report
    assert "indexer" not in report
    assert "Indexer" not in report
    assert "Q:KV ratio" not in report
    assert "2 (K + V)" not in report


def test_deepseek_v32_renders_indexer_rows_and_cache_term(
    v32_config: DeepseekV32Config,
) -> None:
    family = resolve_family("deepseek-ai/DeepSeek-V3.2-Exp", v32_config)
    assert isinstance(family, DeepseekV32Family)

    analysis = family.analyze(Workload(context=8_192, batch_size=1))
    report = render_analysis(analysis, explain=True)

    assert analysis.params.total == V32_PARAMS
    assert analysis.memory.kv_bytes_per_token == V32_KV_BYTES
    assert "~671.88B" in report
    assert "Type                DSA (MLA)" in report
    assert "Layout              Compressed MLA + indexer" in report
    assert "Indexer heads       64" in report
    assert "Indexer key dim     128" in report
    assert "Selected tokens     up to 2,048 per query" in report
    assert "Per token           83.88 KiB" in report
    assert (
        "61 layers × (512 KV latent + 64 RoPE key + 128 indexer key) × 2 bytes (BF16)"
        in report
    )
    assert "The indexer uses 64 query heads and one shared key per token." in report
    assert "does not cap cache length" in report
    assert "Q:KV ratio" not in report


def test_deepseek_v32_cache_is_not_capped_by_top_k(
    v32_config: DeepseekV32Config,
) -> None:
    family = DeepseekV32Family("deepseek-ai/DeepSeek-V3.2-Exp", v32_config)
    top_k = family.spec.attention.indexer.top_k

    short = family.analyze(Workload(context=top_k))
    long = family.analyze(Workload(context=4 * top_k))

    assert long.memory.kv_bytes_per_sequence == 4 * short.memory.kv_bytes_per_sequence
