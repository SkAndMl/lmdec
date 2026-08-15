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


def _walk_fsm(
    state: int,
    fsm: FSM,
    text: str,
) -> int | None:

    for ch in text:
        symbol = fsm.alphabet[ch]
        if symbol not in fsm.map[state]:
            return None

        state = fsm.map[state][symbol]

    return state


def _mask_logits(
    fsm: FSM,
    current_fsm_state: int,
    logits: torch.Tensor,
    id_to_str: dict[int, str],
    eos_token_id: int,
) -> tuple[torch.Tensor, dict[int, int]]:

    valid_ids: dict[int, int] = {}
    for token_id, token in id_to_str.items():
        if (next_state := _walk_fsm(current_fsm_state, fsm, token)) is not None:
            valid_ids[token_id] = next_state

    if current_fsm_state in fsm.finals:
        valid_ids[eos_token_id] = current_fsm_state

    mask = torch.zeros_like(logits, dtype=bool)
    mask[:, list(valid_ids.keys())] = 1
    masked_logits = logits.masked_fill(~mask, -float("inf"))
    return masked_logits, valid_ids


def regex_generate(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    regex: str,
    max_new_tokens: int = 64,
) -> str:

    if not hasattr(tokenizer, "eos_token_id"):
        raise TypeError("tokenizer does not have 'eos_token_id'")

    model_inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    generated = ""

    id_to_str = _build_id_to_str(tokenizer)
    fsm = interegular.parse_pattern(regex).to_fsm()
    current_fsm_state = fsm.initial

    for _ in range(max_new_tokens):
        with torch.inference_mode():
            outputs = model(**model_inputs)
            logits: torch.Tensor = outputs.logits
            next_token_logits = logits[:, -1, :]

            masked_logits, valid_ids = _mask_logits(
                fsm=fsm,
                current_fsm_state=current_fsm_state,
                logits=next_token_logits,
                id_to_str=id_to_str,
                eos_token_id=tokenizer.eos_token_id,
            )
            next_tokens = masked_logits.argmax(dim=-1, keepdim=True)

            next_token = next_tokens[0, 0].item()
            if next_token == tokenizer.eos_token_id:
                break

            current_fsm_state = valid_ids[next_token]
            generated += id_to_str[next_token]

            if current_fsm_state in fsm.finals and len(fsm.map[current_fsm_state]) == 0:
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
