import regex
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _build_id_to_str(tokenizer: AutoTokenizer) -> dict[int, str]:
    id_to_str: dict[int, str] = {}
    for token_id in range(len(tokenizer)):
        raw = tokenizer.convert_ids_to_tokens(token_id)
        id_to_str[token_id] = tokenizer.convert_tokens_to_string([raw])

    return id_to_str


def _mask_logits(
    generated: str,
    logits: torch.Tensor,
    regex_pattern: regex.Pattern,
    id_to_str: dict[int, str],
) -> torch.Tensor:

    valid_ids = []
    for token_id, token in id_to_str.items():
        m = regex_pattern.fullmatch(generated + token, partial=True)
        if m is None:
            continue

        valid_ids.append(token_id)

    mask = torch.zeros_like(logits, dtype=bool)
    mask[:, valid_ids] = 1
    masked_logits = logits.masked_fill(~mask, -float("inf"))
    return masked_logits


def constrained_generation(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    regex_pattern: regex.Pattern,
) -> str:

    if not hasattr(tokenizer, "eos_token_id"):
        raise TypeError("tokenizer does not have 'eos_token_id'")

    model_inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    generated = ""

    id_to_str = _build_id_to_str(tokenizer)

    while True:
        with torch.inference_mode():
            outputs = model(**model_inputs)
            logits: torch.Tensor = outputs.logits
            next_token_logits = logits[:, -1, :]
            masked_logits = _mask_logits(
                generated=generated,
                logits=next_token_logits,
                regex_pattern=regex_pattern,
                id_to_str=id_to_str,
            )
            next_tokens = masked_logits.argmax(dim=-1, keepdim=True)

            next_token = next_tokens[0, 0].item()
            if next_token == tokenizer.eos_token_id:
                break

            generated += tokenizer.decode([next_token])

            if regex_pattern.fullmatch(generated):
                break

            model_inputs["input_ids"] = torch.cat(
                (
                    model_inputs["input_ids"],
                    next_tokens,
                ),
                dim=-1,
            )
            model_inputs["attention_mask"] = torch.ones_like(
                model_inputs["input_ids"], device=model.device
            )

    return generated
