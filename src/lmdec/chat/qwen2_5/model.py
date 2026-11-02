from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.attention.flex_attention import (
    BlockMask,
    _mask_mod_signature,
    flex_attention,
)


def find_multiple(n, k):
    if n % k == 0:
        return n

    return n + k - (n % k)


def get_mask_mod(mask_mod: _mask_mod_signature, offset: int):
    def _mask_mod(b, h, q, k):
        return mask_mod(b, h, q + offset, k)

    return _mask_mod


@dataclass
class ModelConfig:
    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"
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


class KVCache(nn.Module):
    def __init__(
        self,
        max_batch_size: int,
        max_seq_length: int,
        n_heads: int,
        head_dim: int,
        dtype: torch.dtype,
    ):
        super().__init__()
        cache_size = (max_batch_size, n_heads, max_seq_length, head_dim)
        self.register_buffer("k_cache", torch.zeros(cache_size, dtype=dtype))
        self.register_buffer("v_cache", torch.zeros(cache_size, dtype=dtype))

    def update(
        self, input_pos: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:

        assert input_pos.shape[0] == k.shape[2], (
            f"input_pos={input_pos.shape}, k={k.shape}"
        )

        self.k_cache.index_copy_(2, input_pos, k)
        self.v_cache.index_copy_(2, input_pos, v)

        return self.k_cache, self.v_cache


def rotate_half(x):
    d = x.shape[-1]

    x1 = x[..., : d // 2]
    x2 = x[..., d // 2 :]

    return torch.cat((-x2, x1), dim=-1)


def fuse_qkv_weights(cfg: ModelConfig, state_dict: dict[str, Any]):
    for i in range(cfg.num_layers):
        prefix = f"layers.{i}.self_attn."
        wt_keys = [f"{prefix}{x}_proj.weight" for x in ["q", "k", "v"]]
        state_dict[f"layers.{i}.self_attn.qkv_proj.weight"] = torch.cat(
            [state_dict[wt_key] for wt_key in wt_keys],
            dim=0,
        )

        for wt_key in wt_keys:
            state_dict.pop(wt_key)

        if cfg.attn_bias:
            bias_keys = [f"{prefix}{x}_proj.bias" for x in ["q", "k", "v"]]
            state_dict[f"layers.{i}.self_attn.qkv_proj.bias"] = torch.cat(
                [state_dict[bias_key] for bias_key in bias_keys],
                dim=0,
            )

            for bias_key in bias_keys:
                state_dict.pop(bias_key)


def fuse_mlp_weights(cfg: ModelConfig, state_dict: dict[str, Any]):
    for i in range(cfg.num_layers):
        prefix = f"layers.{i}.mlp."
        wt_keys = [f"{prefix}{x}_proj.weight" for x in ["gate", "up"]]
        state_dict[f"layers.{i}.mlp.gate_up_proj.weight"] = torch.cat(
            [state_dict[wt_key] for wt_key in wt_keys],
            dim=0,
        )

        for wt_key in wt_keys:
            state_dict.pop(wt_key)

        if cfg.mlp_bias:
            bias_keys = [f"{prefix}{x}_proj.bias" for x in ["gate", "up"]]
            state_dict[f"layers.{i}.mlp.gate_up_proj.bias"] = torch.cat(
                [state_dict[bias_key] for bias_key in bias_keys],
                dim=0,
            )

            for bias_key in bias_keys:
                state_dict.pop(bias_key)


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


def apply_rope(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
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

        self.qkv_proj = nn.Linear(
            cfg.model_dim,
            self.head_dim * (self.attn_heads + 2 * self.kv_heads),
            bias=cfg.attn_bias,
        )

        self.o_proj = nn.Linear(
            self.head_dim * self.attn_heads, cfg.model_dim, bias=False
        )

        self.kv_cache: KVCache | None = None

    def forward(
        self,
        *,
        x: Tensor,
        mask: BlockMask,
        position_ids: Tensor,
        position_embeddings: Tensor,
    ) -> Tensor:

        b, t, _ = x.shape

        qkv: Tensor = self.qkv_proj(x)  # b, t, head_dim * (q_h + 2*kv_h)
        q, k, v = qkv.split(
            split_size=[
                self.head_dim * self.attn_heads,
                self.head_dim * self.kv_heads,
                self.head_dim * self.kv_heads,
            ],
            dim=2,
        )

        q = q.view(b, t, self.attn_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        q, k = apply_rope(q, k, cos, sin)

        if self.kv_cache is not None:
            k, v = self.kv_cache.update(position_ids, k, v)

        out: Tensor = flex_attention(
            q,
            k,
            v,
            block_mask=mask,
            enable_gqa=self.attn_heads != self.kv_heads,
        )

        out = (
            out.transpose(1, 2).contiguous().view(b, t, self.head_dim * self.attn_heads)
        )

        return self.o_proj(out)


class Qwen2MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()

        self.cfg = cfg

        self.gate_up_proj = nn.Linear(
            cfg.model_dim,
            cfg.intermediate_size * 2,
            cfg.mlp_bias,
        )

        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.model_dim, cfg.mlp_bias)
        self.act_fn = F.silu

    def forward(self, x: Tensor) -> Tensor:

        gate_up_fused: Tensor = self.gate_up_proj(x)  # b, t, intermediate_size * 2
        gate, up = gate_up_fused.split(
            split_size=[
                self.cfg.intermediate_size,
                self.cfg.intermediate_size,
            ],
            dim=2,
        )

        return self.down_proj(self.act_fn(gate) * up)


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
        mask: BlockMask,
        position_ids: Tensor,
        position_embeddings: Tensor,
    ) -> Tensor:
        attn_out = self.self_attn(
            x=self.input_layernorm(x),
            mask=mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
        )
        x = x + attn_out
        x = x + self.mlp(self.post_attention_layernorm(x))

        return x


class Qwen2Model(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()

        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.model_dim)
        self.layers = nn.ModuleList(
            [Qwen2DecoderLayer(cfg) for _ in range(cfg.num_layers)]
        )
        self.norm = Qwen2RMSNorm(cfg)
        self.rotary_emb = Qwen2RotaryEmbedding(cfg)

        self.lm_head = nn.Linear(
            cfg.model_dim,
            cfg.vocab_size,
            bias=False,
            dtype=cfg.dtype,
        )

        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        self.max_batch_size = -1
        self.max_seq_length = -1
        self.get_mask_mod = get_mask_mod

    def setup_caches(self, max_batch_size: int, max_seq_length: int):
        head_dim = self.cfg.model_dim // self.cfg.num_attention_heads
        self.max_seq_length = find_multiple(max_seq_length, 8)
        self.max_batch_size = max_batch_size
        dtype = self.lm_head.weight.dtype

        for layer in self.layers:
            layer.self_attn.kv_cache = KVCache(
                max_batch_size=self.max_batch_size,
                max_seq_length=self.max_seq_length,
                n_heads=self.cfg.num_key_value_heads,
                head_dim=head_dim,
                dtype=dtype,
            ).to(self.lm_head.weight.device)

    def forward(
        self,
        *,
        input_ids: Tensor,
        position_ids: Tensor,
        mask: BlockMask,
    ) -> Tensor:

        x = self.embed_tokens(input_ids)
        position_embeddings = self.rotary_emb(x, position_ids)

        mask.mask_mod = self.get_mask_mod(mask.mask_mod, position_ids[0])

        for layer in self.layers:
            x = layer(
                x=x,
                mask=mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
            )

        return self.lm_head(self.norm(x))

    @staticmethod
    def from_pretrained(cfg: ModelConfig) -> "Qwen2Model":
        from transformers import AutoModelForCausalLM

        model = Qwen2Model(cfg)
        state_dict = AutoModelForCausalLM.from_pretrained(cfg.model_id).state_dict()

        for k in list(state_dict.keys()):
            if k.startswith("model."):
                state_dict[k[len("model.") :]] = state_dict.pop(k)

        fuse_qkv_weights(cfg, state_dict)
        fuse_mlp_weights(cfg, state_dict)

        model.load_state_dict(state_dict)

        return model
