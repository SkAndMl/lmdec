import logging

import torch
import torch._logging
from transformers import AutoTokenizer

from lmdec.chat.qwen2_5 import KVCache, ModelConfig, Qwen2ForCausalLM

system_prompt = "You are a friendly chat assistant. Reply under 3 lines"

torch._logging.set_logs(
    dynamo=logging.ERROR,
    inductor=logging.ERROR,
    aot=logging.ERROR,
)


def chat():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    cfg = ModelConfig("Qwen/Qwen2.5-0.5B-Instruct")
    model = Qwen2ForCausalLM.from_pretrained(cfg).to(device)

    model._decode = torch.compile(model._decode, mode="reduce-overhead")

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
        torch._dynamo.mark_static(c.k)
        torch._dynamo.mark_static(c.v)

    chat_history = [{"role": "system", "content": system_prompt}]

    cached_len = 0

    while True:
        user_message = input("User: ").strip()
        if not user_message:
            continue
        if user_message == "/exit":
            break

        if user_message == "/reset":
            cached_len = 0
            chat_history = [{"role": "system", "content": system_prompt}]
            print("Context cleared")
            continue

        chat_history.append({"role": "user", "content": user_message})

        tokens = tokenizer.apply_chat_template(
            conversation=chat_history,
            add_generation_prompt=True,
            tokenize=True,
        )["input_ids"]
        input_token_ids = torch.tensor([tokens], dtype=torch.long, device=device)

        output = model.generate(
            input_ids=input_token_ids,
            kv_cache=kv_cache,
            max_tokens=1024,
            cached_len=cached_len,
        )

        cached_len = len(tokens)

        response = tokenizer.decode(
            output["output_tokens"],
            skip_special_tokens=True,
        )

        print(f"Assistant: {response}")
        print(
            f"stats - num_output_tokens: {len(output['output_tokens'])}; ttft: {output['ttft']:.4f};"
            f" tpot: {output['tpot']:.4f}; total_time: {output['total_time']:.4f}"
        )

        chat_history.append({"role": "assistant", "content": response})

        cached_len += len(output["output_tokens"]) - 1


if __name__ == "__main__":
    chat()
