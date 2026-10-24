import torch
from transformers import AutoTokenizer

from lmdec.chat.qwen2_5 import ModelConfig, Qwen2ForCausalLM

system_prompt = "You are a friendly chat assistant. Reply under 3 lines"


def chat():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    model = Qwen2ForCausalLM.from_pretrained(
        cfg=ModelConfig("Qwen/Qwen2.5-0.5B-Instruct")
    ).to(device)

    chat_history = [{"role": "system", "content": system_prompt}]

    while True:
        user_message = input("User: ").strip()
        if not user_message:
            continue
        if user_message == "/exit":
            break

        chat_history.append({"role": "user", "content": user_message})

        tokens = tokenizer.apply_chat_template(
            conversation=chat_history,
            add_generation_prompt=True,
            tokenize=True,
        )["input_ids"]
        input_token_ids = torch.tensor([tokens], dtype=torch.long, device=device)

        output = model.generate(input_token_ids, max_tokens=256)

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


if __name__ == "__main__":
    chat()
