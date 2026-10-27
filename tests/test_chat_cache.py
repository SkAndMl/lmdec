import pytest
import torch

from lmdec.chat.qwen2_5 import KVCache, ModelConfig, Qwen2ForCausalLM


@pytest.fixture
def model_and_cache():
    torch.manual_seed(42)
    cfg = ModelConfig(
        vocab_size=64,
        max_position_embeddings=64,
        model_dim=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=64,
        num_layers=2,
        dtype=torch.float32,
    )
    model = Qwen2ForCausalLM(cfg).eval()

    def new_cache():
        shape = (
            1,
            cfg.num_key_value_heads,
            64,
            cfg.model_dim // cfg.num_attention_heads,
        )
        return {
            layer: KVCache(k=torch.zeros(shape), v=torch.zeros(shape))
            for layer in range(cfg.num_layers)
        }

    return model, new_cache


@pytest.mark.parametrize("split", [1, 3, 5])
def test_cached_prefill_matches_full_prefill(model_and_cache, split):
    model, new_cache = model_and_cache
    tokens = torch.tensor([[1, 7, 12, 4, 23, 9]])
    positions = torch.arange(tokens.shape[1]).unsqueeze(0)

    with torch.inference_mode():
        fresh_cache = new_cache()
        full_logits = model._prefill(tokens, positions, fresh_cache)
        cached = new_cache()
        model._prefill(tokens[:, :split], positions[:, :split], cached)
        cached_logits = model._prefill(tokens[:, split:], positions[:, split:], cached)

    torch.testing.assert_close(
        cached_logits, full_logits[:, split:], rtol=1e-5, atol=1e-6
    )
    for layer in fresh_cache:
        torch.testing.assert_close(
            cached[layer].k, fresh_cache[layer].k, rtol=1e-5, atol=1e-6
        )
        torch.testing.assert_close(
            cached[layer].v, fresh_cache[layer].v, rtol=1e-5, atol=1e-6
        )


@pytest.mark.parametrize("first_max_tokens", [1, 4])
def test_next_turn_matches_generation_with_fresh_cache(
    model_and_cache, first_max_tokens
):
    model, new_cache = model_and_cache
    cache = new_cache()
    prompt = torch.tensor([[1, 7, 12, 4]])
    first = model.generate(
        prompt, cache, cached_len=0, max_tokens=first_max_tokens, eot_token_ids=set()
    )

    # Model the engine's next formatted turn, keeping the cached prefix exact.
    conversation = torch.tensor(
        [prompt[0].tolist() + first["output_tokens"] + [5, 6, 8]]
    )
    cached_len = prompt.shape[1] + len(first["output_tokens"]) - 1
    reused = model.generate(
        conversation, cache, cached_len=cached_len, max_tokens=4, eot_token_ids=set()
    )
    fresh = model.generate(
        conversation, new_cache(), cached_len=0, max_tokens=4, eot_token_ids=set()
    )

    assert reused["output_tokens"] == fresh["output_tokens"]


def test_next_turn_matches_after_end_of_turn(model_and_cache):
    model, new_cache = model_and_cache
    prompt = torch.tensor([[1, 7, 12, 4]])
    # Discover the second greedy token, then use it to force an early stop.
    baseline = model.generate(
        prompt, new_cache(), cached_len=0, max_tokens=2, eot_token_ids=set()
    )
    cache = new_cache()
    first = model.generate(
        prompt,
        cache,
        cached_len=0,
        max_tokens=4,
        eot_token_ids={baseline["output_tokens"][1]},
    )
    assert first["output_tokens"] == baseline["output_tokens"][:1]

    conversation = torch.tensor(
        [prompt[0].tolist() + first["output_tokens"] + [5, 6, 8]]
    )
    cached_len = prompt.shape[1] + len(first["output_tokens"]) - 1
    reused = model.generate(
        conversation, cache, cached_len=cached_len, max_tokens=4, eot_token_ids=set()
    )
    fresh = model.generate(
        conversation, new_cache(), cached_len=0, max_tokens=4, eot_token_ids=set()
    )

    assert reused["output_tokens"] == fresh["output_tokens"]
