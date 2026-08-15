import interegular
import torch
from interegular import FSM
from transformers import AutoModelForCausalLM, AutoTokenizer


def _build_id_to_str(tokenizer: AutoTokenizer) -> dict[int, str]:
    id_to_str: dict[int, str] = {}
    for token_id in range(len(tokenizer)):
        raw = tokenizer.convert_ids_to_tokens(token_id)
        id_to_str[token_id] = tokenizer.convert_tokens_to_string([raw])

    return id_to_str


def _mask_logits(
    generated: str,
    fsm: FSM,
    logits: torch.Tensor,
    id_to_str: dict[int, str],
    eot_id: int,
) -> torch.Tensor:

    def _is_valid_token(token: str) -> bool:
        text = generated + token
        state = fsm.initial
        for ch in text:
            symbol = fsm.alphabet[ch]
            if symbol not in fsm.map[state]:
                return False
            state = fsm.map[state][symbol]

        return True

    valid_ids = [eot_id]
    for token_id, token in id_to_str.items():
        if _is_valid_token(token):
            valid_ids.append(token_id)

    mask = torch.zeros_like(logits, dtype=bool)
    mask[:, valid_ids] = 1
    masked_logits = logits.masked_fill(~mask, -float("inf"))
    return masked_logits


def regex_generate(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    regex: str,
) -> str:

    if not hasattr(tokenizer, "eos_token_id"):
        raise TypeError("tokenizer does not have 'eos_token_id'")

    model_inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    generated = ""

    id_to_str = _build_id_to_str(tokenizer)
    fsm = interegular.parse_pattern(regex).to_fsm()

    while True:
        with torch.inference_mode():
            outputs = model(**model_inputs)
            logits: torch.Tensor = outputs.logits
            next_token_logits = logits[:, -1, :]
            masked_logits = _mask_logits(
                generated=generated,
                fsm=fsm,
                logits=next_token_logits,
                id_to_str=id_to_str,
                eot_id=tokenizer.eos_token_id,
            )
            next_tokens = masked_logits.argmax(dim=-1, keepdim=True)

            next_token = next_tokens[0, 0].item()
            if next_token == tokenizer.eos_token_id:
                break

            generated += id_to_str[next_token]

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
