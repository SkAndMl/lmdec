from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class ModelConfig:
    model_id: str = "Qwen/Qwen2.5-0.5B"
    vocab_size: int = 151936
    max_position_embeddings: int = 32768
    model_dim: int = 896
    num_attention_heads: int = 14
    num_key_value_heads: int = 2
    intermediate_size: int = 4864
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    tie_word_embeddings: bool = True
    attn_bias: bool = True
    mlp_bias: bool = False
    num_layers: int = 24
    dtype: torch.dtype = torch.float16


@dataclass
class KVCache:
    k: torch.Tensor
    v: torch.Tensor
    prefill: bool


def rotate_half(x):
    d = x.shape[-1]

    x1 = x[..., : d // 2]
    x2 = x[..., d // 2 :]

    return torch.cat((-x2, x1), dim=-1)


class Qwen2RotaryEmbedding(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()

        head_dim = cfg.model_dim // cfg.num_attention_heads
        inv_freq = 1.0 / (
            cfg.rope_theta
            ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x: Tensor, position_ids: Tensor) -> tuple[Tensor, Tensor]:
        # position_ids: [B, T]
        inv_freq = self.inv_freq[None, None, :].float()
        device_type = x.device.type if x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            angles = position_ids[..., None].float() * inv_freq
            angles = torch.cat((angles, angles), dim=-1)
            cos, sin = angles.cos(), angles.sin()

        return cos.to(x.dtype), sin.to(x.dtype)


def apply_rope(q, k, cos, sin):
    # q: [B, H, T, D]
    # k: [B, KV_H, T, D]
    # cos/sin: [B, T, D]

    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)

    q = q * cos + rotate_half(q) * sin
    k = k * cos + rotate_half(k) * sin

    return q, k


class Qwen2RMSNorm(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()

        self.weight = nn.Parameter(torch.ones(size=(cfg.model_dim,)))
        self.eps = cfg.rms_norm_eps

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

        return self.weight * x.to(dtype)


class Qwen2Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()

        assert cfg.num_attention_heads % cfg.num_key_value_heads == 0

        self.attn_heads = cfg.num_attention_heads
        self.kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.model_dim // self.attn_heads

        self.q_proj = nn.Linear(
            cfg.model_dim, self.head_dim * self.attn_heads, bias=cfg.attn_bias
        )
        self.k_proj = nn.Linear(
            cfg.model_dim, self.head_dim * self.kv_heads, bias=cfg.attn_bias
        )
        self.v_proj = nn.Linear(
            cfg.model_dim, self.head_dim * self.kv_heads, bias=cfg.attn_bias
        )
        self.o_proj = nn.Linear(
            self.head_dim * self.attn_heads, cfg.model_dim, bias=False
        )

    def forward(
        self,
        *,
        x: Tensor,
        mask: Tensor | None = None,
        position_embeddings: Tensor | None = None,
        lengths: torch.Tensor | None = None,  # b,
        kv_cache: KVCache | None = None,
    ) -> Tensor:
        b, t, _ = x.shape

        q: Tensor = (
            self.q_proj(x).view(b, t, self.attn_heads, self.head_dim).transpose(1, 2)
        )  # b, q_h, t, head_dim
        curr_k: Tensor = (
            self.k_proj(x).view(b, t, self.kv_heads, self.head_dim).transpose(1, 2)
        )  # b, kv_h, t, head_dim
        curr_v: Tensor = (
            self.v_proj(x).view(b, t, self.kv_heads, self.head_dim).transpose(1, 2)
        )  # b, kv_h, t, head_dim

        cos, sin = position_embeddings
        q, curr_k = apply_rope(q, curr_k, cos, sin)

        k, v = curr_k, curr_v
        if kv_cache is not None:
            if lengths is None:
                raise RuntimeError("kv_cache is not None but lengths is")

            if kv_cache.prefill:
                kv_cache.k[:, :, :lengths, :] = k
                kv_cache.v[:, :, :lengths, :] = v
            else:
                kv_cache.k[:, :, lengths - 1 : lengths, :] = k
                kv_cache.v[:, :, lengths - 1 : lengths, :] = v

            k, v = kv_cache.k[:, :, :lengths, :], kv_cache.v[:, :, :lengths, :]

        kv_repeats = self.attn_heads // self.kv_heads

        attn_k = k.repeat_interleave(kv_repeats, dim=1)
        attn_v = v.repeat_interleave(kv_repeats, dim=1)

        attn_scores: Tensor = (
            q @ attn_k.transpose(2, 3) / (self.head_dim**0.5)
        )  # b, q_h, t, t
        attention_mask = torch.ones(t, t, device=x.device, dtype=torch.bool).triu(1)
        if mask is not None:
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
            attention_mask = attention_mask | mask
        attn_scores.masked_fill_(attention_mask, value=float("-inf"))

        attn_weights = F.softmax(
            attn_scores,
            dim=-1,
            dtype=torch.float32,
        ).to(q.dtype)
        out = (
            (attn_weights @ attn_v)
            .transpose(1, 2)
            .contiguous()
            .view(b, t, self.head_dim * self.attn_heads)
        )

        return self.o_proj(out)


class Qwen2MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()

        self.gate_proj = nn.Linear(cfg.model_dim, cfg.intermediate_size, cfg.mlp_bias)
        self.up_proj = nn.Linear(cfg.model_dim, cfg.intermediate_size, cfg.mlp_bias)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.model_dim, cfg.mlp_bias)
        self.act_fn = F.silu

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Qwen2DecoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()

        self.self_attn = Qwen2Attention(cfg)
        self.mlp = Qwen2MLP(cfg)
        self.input_layernorm = Qwen2RMSNorm(cfg)
        self.post_attention_layernorm = Qwen2RMSNorm(cfg)

    def forward(
        self,
        *,
        x: Tensor,
        mask: Tensor | None = None,
        position_embeddings: Tensor | None = None,
        kv_cache: KVCache | None = None,
        lengths: torch.Tensor | None = None,
    ) -> Tensor:
        attn_out = self.self_attn(
            x=self.input_layernorm(x),
            mask=mask,
            position_embeddings=position_embeddings,
            kv_cache=kv_cache,
            lengths=lengths,
        )
        x = x + attn_out
        x = x + self.mlp(self.post_attention_layernorm(x))

        return x


class Qwen2Model(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()

        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.model_dim)
        self.layers = nn.ModuleList(
            [Qwen2DecoderLayer(cfg) for _ in range(cfg.num_layers)]
        )
        self.norm = Qwen2RMSNorm(cfg)
        self.rotary_emb = Qwen2RotaryEmbedding(cfg)

    def forward(
        self,
        *,
        input_ids: Tensor,
        position_ids: Tensor | None = None,
        mask: Tensor | None = None,
        kv_cache: dict[int, KVCache] | None = None,
        lengths: torch.Tensor | None = None,
    ) -> Tensor:

        if position_ids is None:
            position_ids = torch.arange(input_ids.shape[1], device=input_ids.device)[
                None, :
            ]

        x = self.embed_tokens(input_ids)
        position_embeddings = self.rotary_emb(x, position_ids)

        for i, layer in enumerate(self.layers):
            layer_kv_cache = None
            if kv_cache is not None:
                layer_kv_cache = kv_cache[i]
            x = layer(
                x=x,
                mask=mask,
                position_embeddings=position_embeddings,
                kv_cache=layer_kv_cache,
                lengths=lengths,
            )

        return self.norm(x)


class Qwen2ForCausalLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()

        self.cfg = cfg
        self.model = Qwen2Model(cfg).to(cfg.dtype)
        self.lm_head = nn.Linear(
            cfg.model_dim,
            cfg.vocab_size,
            bias=False,
            dtype=cfg.dtype,
        )

        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor | None = None,
        mask: Tensor | None = None,
        kv_cache: dict[int, KVCache] | None = None,
        lengths: torch.Tensor | None = None,
    ) -> Tensor:
        x = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            mask=mask,
            kv_cache=kv_cache,
            lengths=lengths,
        )
        return self.lm_head(x)

    def _prefill(
        self,
        x: Tensor,
        position_ids: Tensor | None = None,
        lengths: Tensor | None = None,
    ) -> tuple[Tensor, dict[int, KVCache]]:

        b, t = x.shape
        if t > self.cfg.max_position_embeddings:
            raise ValueError()

        cache_length = t + 1024
        cache_tensor_shape = (
            b,
            self.cfg.num_key_value_heads,
            cache_length,
            self.cfg.model_dim // self.cfg.num_attention_heads,
        )

        kv_cache = {
            layer: KVCache(
                k=torch.empty(
                    size=cache_tensor_shape, device=x.device, dtype=self.cfg.dtype
                ),
                v=torch.empty(
                    size=cache_tensor_shape, device=x.device, dtype=self.cfg.dtype
                ),
                prefill=True,
            )
            for layer in range(self.cfg.num_layers)
        }

        logits = self(
            input_ids=x,
            position_ids=position_ids,
            mask=None,
            kv_cache=kv_cache,
            lengths=lengths,
        )

        for layer in range(self.cfg.num_layers):
            kv_cache[layer].prefill = False

        return logits, kv_cache

    @torch.inference_mode()
    def generate(self, input_ids: Tensor, max_tokens: int = 20) -> Tensor:

        if max_tokens == 0:
            return input_ids

        assert len(input_ids) == 1
        x = input_ids.clone()
        position_ids = torch.arange(
            x.shape[-1], dtype=torch.long, device=x.device
        ).unsqueeze(0)

        lengths = torch.tensor((x.shape[-1],), device=x.device, dtype=torch.long)

        logits, kv_cache = self._prefill(
            x,
            position_ids,
            lengths=lengths,
        )

        next_token_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        x = torch.cat([x, next_token_id], dim=-1).to(x.device)
        position_ids = torch.tensor(
            [[x.shape[-1] - 1]],
            dtype=torch.long,
            device=x.device,
        )
        lengths += 1

        for _ in range(max_tokens - 1):
            logits = self(
                input_ids=x[:, -1:],
                position_ids=position_ids,
                mask=None,
                kv_cache=kv_cache,
                lengths=lengths,
            )

            next_token_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            x = torch.cat([x, next_token_id], dim=-1).to(x.device)
            position_ids = torch.tensor(
                [[x.shape[-1] - 1]], dtype=torch.long, device=x.device
            )
            lengths += 1

        return x

    @staticmethod
    @torch.inference_mode()
    def check(cfg: ModelConfig):
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            PreTrainedTokenizerBase,
        )

        tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(cfg.model_id)

        hf = AutoModelForCausalLM.from_pretrained(
            cfg.model_id,
            attn_implementation="eager",
        )
        ours = Qwen2ForCausalLM.from_pretrained(cfg)

        hf.eval()
        ours.eval()

        input_ids: Tensor = tokenizer(
            "The capital of France is",
            return_tensors="pt",
        ).input_ids

        hf_logits: Tensor = hf(input_ids).logits
        our_logits = ours(input_ids)

        print("max diff:", (hf_logits - our_logits).abs().max())
        print(
            "HF:",
            hf_logits[:, -1].argmax(-1),
            "ours:",
            our_logits[:, -1].argmax(-1),
        )

    @staticmethod
    def from_pretrained(cfg: ModelConfig) -> "Qwen2ForCausalLM":
        from transformers import AutoModelForCausalLM

        model = Qwen2ForCausalLM(cfg)
        hf_model_state_dict = AutoModelForCausalLM.from_pretrained(
            cfg.model_id
        ).state_dict()

        model.load_state_dict(hf_model_state_dict)

        return model


if __name__ == "__main__":
    import time

    from transformers import AutoTokenizer, PreTrainedTokenizerBase

    model_id = "Qwen/Qwen2.5-0.5B"
    num_tokens = 20

    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(model_id)
    cfg = ModelConfig(dtype=torch.float32)

    # Qwen2ForCausalLM.check(cfg)

    model = Qwen2ForCausalLM.from_pretrained(cfg)

    text = "Hello"
    tokens = tokenizer.encode(text)
    input_ids = torch.tensor([tokens], dtype=torch.long)

    start = time.perf_counter()

    generation = model.generate(input_ids, num_tokens)

    total = time.perf_counter() - start

    output_text = tokenizer.decode(generation[0].tolist())
    print(f"Generated: {output_text}")

    print(f"time: {total:.4f}")
    print(f"tokens/sec: {num_tokens / total:.4f}")
