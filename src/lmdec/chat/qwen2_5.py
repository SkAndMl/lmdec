import os
import time
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.profiler import ProfilerActivity, profile


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


def rotate_half(x):
    d = x.shape[-1]

    x1 = x[..., : d // 2]
    x2 = x[..., d // 2 :]

    return torch.cat((-x2, x1), dim=-1)


def fuse_qkv_weights(cfg: ModelConfig, state_dict: dict[str, Any]):
    for i in range(cfg.num_layers):
        prefix = f"model.layers.{i}.self_attn."
        wt_keys = [f"{prefix}{x}_proj.weight" for x in ["q", "k", "v"]]
        state_dict[f"model.layers.{i}.self_attn.qkv_proj.weight"] = torch.cat(
            [state_dict[wt_key] for wt_key in wt_keys],
            dim=0,
        )

        for wt_key in wt_keys:
            state_dict.pop(wt_key)

        if cfg.attn_bias:
            bias_keys = [f"{prefix}{x}_proj.bias" for x in ["q", "k", "v"]]
            state_dict[f"model.layers.{i}.self_attn.qkv_proj.bias"] = torch.cat(
                [state_dict[bias_key] for bias_key in bias_keys],
                dim=0,
            )

            for bias_key in bias_keys:
                state_dict.pop(bias_key)


def fuse_mlp_weights(cfg: ModelConfig, state_dict: dict[str, Any]):
    for i in range(cfg.num_layers):
        prefix = f"model.layers.{i}.mlp."
        wt_keys = [f"{prefix}{x}_proj.weight" for x in ["gate", "up"]]
        state_dict[f"model.layers.{i}.mlp.gate_up_proj.weight"] = torch.cat(
            [state_dict[wt_key] for wt_key in wt_keys],
            dim=0,
        )

        for wt_key in wt_keys:
            state_dict.pop(wt_key)

        if cfg.mlp_bias:
            bias_keys = [f"{prefix}{x}_proj.bias" for x in ["gate", "up"]]
            state_dict[f"model.layers.{i}.mlp.gate_up_proj.bias"] = torch.cat(
                [state_dict[bias_key] for bias_key in bias_keys],
                dim=0,
            )

            for bias_key in bias_keys:
                state_dict.pop(bias_key)


def build_mask(position_ids: Tensor, cache_len: int) -> Tensor:
    # position_ids -> 1, T
    slots = torch.arange(cache_len, device=position_ids.device)  # S
    mask = slots[None, :] <= position_ids[0][:, None]  # T, S
    return mask[None, None]


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

        if (
            not torch.compiler.is_compiling()
            and x.is_cuda
            and x.dtype == torch.float16
            and x.is_contiguous()
        ):
            from lmdec.chat.kernels.qwen2_5_kernels import rmsnorm_triton

            return rmsnorm_triton(
                x,
                self.weight,
                self.eps,
            )

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

    def forward(
        self,
        *,
        x: Tensor,
        attention_mask: Tensor,
        position_ids: Tensor,
        position_embeddings: Tensor,
        kv_cache: KVCache,
    ) -> Tensor:

        b, t, _ = x.shape

        qkv: Tensor = self.qkv_proj(x)  # b, t, head_dim * (q_h + 2*kv_h)
        q, curr_k, curr_v = qkv.split(
            split_size=[
                self.head_dim * self.attn_heads,
                self.head_dim * self.kv_heads,
                self.head_dim * self.kv_heads,
            ],
            dim=2,
        )

        q = q.view(b, t, self.attn_heads, self.head_dim).transpose(
            1, 2
        )  # b, q_h, t, head_dim
        curr_k = curr_k.view(b, t, self.kv_heads, self.head_dim).transpose(
            1, 2
        )  # b, kv_h, t, head_dim
        curr_v = curr_v.view(b, t, self.kv_heads, self.head_dim).transpose(
            1, 2
        )  # b, kv_h, t, head_dim

        cos, sin = position_embeddings
        q, curr_k = apply_rope(q, curr_k, cos, sin)

        k, v = curr_k, curr_v

        kv_cache.k.index_copy_(2, position_ids[0], k)
        kv_cache.v.index_copy_(2, position_ids[0], v)

        k, v = kv_cache.k, kv_cache.v

        out = F.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            dropout_p=0.0,
            attn_mask=attention_mask,
            enable_gqa=True,
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

        if (
            self.act_fn.__name__ == "silu"
            and not torch.compiler.is_compiling()
            and x.is_cuda
            and x.dtype == torch.float16
            and x.is_contiguous()
        ):
            from lmdec.chat.kernels.qwen2_5_kernels import swiglu_triton

            act = swiglu_triton(gate, up)
            return self.down_proj(act)

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
        attention_mask: Tensor,
        position_ids: Tensor,
        position_embeddings: Tensor,
        kv_cache: KVCache,
    ) -> Tensor:
        attn_out = self.self_attn(
            x=self.input_layernorm(x),
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            kv_cache=kv_cache,
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
        position_ids: Tensor,
        attention_mask: Tensor,
        kv_cache: dict[int, KVCache],
    ) -> Tensor:

        x = self.embed_tokens(input_ids)
        position_embeddings = self.rotary_emb(x, position_ids)

        for i, layer in enumerate(self.layers):
            x = layer(
                x=x,
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                kv_cache=kv_cache[i],
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
        position_ids: Tensor,
        attention_mask: Tensor,
        kv_cache: dict[int, KVCache],
    ) -> Tensor:
        x = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
        )
        return self.lm_head(x)

    def _prefill(
        self,
        x: Tensor,
        position_ids: Tensor,
        kv_cache: dict[int, KVCache],
    ) -> Tensor:

        attention_mask = build_mask(position_ids, kv_cache[0].k.shape[-2])

        logits = self(
            input_ids=x,
            position_ids=position_ids,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
        )

        return logits

    def _decode(
        self,
        x: Tensor,
        position_ids: Tensor,
        kv_cache: dict[int, KVCache],
    ) -> Tensor:

        attention_mask = build_mask(position_ids, kv_cache[0].k.shape[-2])

        logits: Tensor = self(
            input_ids=x,
            position_ids=position_ids,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
        )

        return logits[:, -1, :].argmax(dim=-1, keepdim=True)

    @torch.inference_mode()
    def generate(
        self,
        input_ids: Tensor,
        kv_cache: dict[int, KVCache],
        cached_len: int,
        max_tokens: int = 20,
        eot_token_ids: set[int] = {151643, 151645},  # noqa: B006
    ) -> dict[str, Any]:

        assert len(input_ids) == 1

        if input_ids.shape[-1] > self.cfg.max_position_embeddings:
            raise ValueError()

        print(f"cached_len: {cached_len}, input_len: {input_ids.shape[-1]}")

        output = {
            "output_tokens": [],
            "ttft": -1,
            "total_time": -1,
            "tpot": -1,
        }

        if max_tokens == 0:
            return output

        x = input_ids.clone()
        position_ids = torch.arange(
            cached_len, x.shape[-1], dtype=torch.long, device=x.device
        ).unsqueeze(0)

        start_time = time.perf_counter()

        logits = self._prefill(
            x[:, cached_len:],
            position_ids,
            kv_cache=kv_cache,
        )

        if x.is_cuda:
            torch.cuda.synchronize()

        output["ttft"] = time.perf_counter() - start_time

        next_token_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        x = torch.cat([x, next_token_id], dim=-1).to(x.device)
        position_ids = torch.tensor(
            [[x.shape[-1] - 1]],
            dtype=torch.long,
            device=x.device,
        )

        for _ in range(max_tokens - 1):
            next_token_id = self._decode(
                x[:, -1:],
                position_ids,
                kv_cache,
            )

            if next_token_id[0, 0].item() in eot_token_ids:
                break

            x = torch.cat([x, next_token_id], dim=-1).to(x.device)
            position_ids += 1

        if x.is_cuda:
            torch.cuda.synchronize()

        output["total_time"] = time.perf_counter() - start_time
        output["output_tokens"] = x[0].tolist()[len(input_ids[0]) :]
        output["tpot"] = (output["total_time"] - output["ttft"]) / len(
            output["output_tokens"]
        )

        return output

    @staticmethod
    def from_pretrained(cfg: ModelConfig) -> "Qwen2ForCausalLM":
        from transformers import AutoModelForCausalLM

        model = Qwen2ForCausalLM(cfg)
        state_dict = AutoModelForCausalLM.from_pretrained(cfg.model_id).state_dict()

        fuse_qkv_weights(cfg, state_dict)
        fuse_mlp_weights(cfg, state_dict)

        model.load_state_dict(state_dict)

        return model


if __name__ == "__main__":
    from transformers import AutoTokenizer, PreTrainedTokenizerBase

    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    num_tokens = 64
    warmup_steps = int(os.environ.get("WARMUP_STEPS", "2"))
    actual_generation = 5

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    dtype = torch.float16
    if device == "cuda" and torch.cuda.is_bf16_supported(including_emulation=False):
        dtype = torch.bfloat16

    print(f"dtype: {dtype}")

    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(model_id)
    cfg = ModelConfig(model_id=model_id, dtype=dtype)

    model = Qwen2ForCausalLM.from_pretrained(cfg)
    model.to(device)

    model._decode = torch.compile(model._decode, mode="reduce-overhead", fullgraph=True)

    cache_length = 4096
    cache_tensor_shape = (
        1,
        cfg.num_key_value_heads,
        cache_length,
        cfg.model_dim // cfg.num_attention_heads,
    )

    kv_cache = {
        layer: KVCache(
            k=torch.zeros(size=cache_tensor_shape, device=device, dtype=cfg.dtype),
            v=torch.zeros(size=cache_tensor_shape, device=device, dtype=cfg.dtype),
        )
        for layer in range(cfg.num_layers)
    }

    for c in kv_cache.values():
        torch._dynamo.mark_static_address(c.k)
        torch._dynamo.mark_static_address(c.v)

    text = "The world is a"
    tokens = tokenizer.encode(text)
    input_ids = torch.tensor([tokens], dtype=torch.long).to(device)

    for _ in range(warmup_steps):
        _ = model.generate(input_ids, kv_cache, num_tokens, eot_token_ids=set())

    torch.cuda.synchronize()

    print("warmup done")

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()

    start.record()
    for _ in range(actual_generation):
        output = model.generate(input_ids, kv_cache, num_tokens, eot_token_ids=set())
    end.record()

    torch.cuda.synchronize()

    total = start.elapsed_time(end)

    generation = tokenizer.decode(output["output_tokens"])

    print(f"Generation: {generation}")
    print(f"time taken: {total / 1000 / actual_generation:.4f} seconds")

    if os.environ.get("PROFILE", "false").lower() in ("1", "true", "yes"):
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            _ = model.generate(input_ids, kv_cache, num_tokens, eot_token_ids=set())
            torch.cuda.synchronize()

        events = prof.key_averages()

        total_gpu = sum(e.self_device_time_total for e in events)

        rows = sorted(events, key=lambda e: e.self_device_time_total, reverse=True)

        print(
            f"{'Name':<60} {'Calls':>6} {'GPU self ms':>12} {'GPU %':>7} {'GPU avg us':>11}"
        )

        for e in rows[:25]:
            print(
                f"{e.key[:60]:<60} "
                f"{e.count:>6} "
                f"{e.self_device_time_total / 1000:>12.3f} "
                f"{100 * e.self_device_time_total / total_gpu:>6.1f}% "
                f"{e.self_device_time_total / e.count:>11.1f}"
            )
