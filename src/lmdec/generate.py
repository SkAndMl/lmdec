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


def _build_state_to_valid_tokens_map(
    fsm: FSM, id_to_str: dict[int, str], eos_token_id: int
) -> dict[int, dict[int, int]]:

    state_to_valid_tokens_map: dict[int, dict[int, int]] = {}

    for state in fsm.states:
        state_to_valid_tokens_map[state] = {}
        for token_id, token in id_to_str.items():
            if (next_state := _walk_fsm(state, fsm, token)) is not None:
                state_to_valid_tokens_map[state][token_id] = next_state

        if state in fsm.finals:
            state_to_valid_tokens_map[state][eos_token_id] = state

    return state_to_valid_tokens_map


def _mask_logits(
    logits: torch.Tensor,
    valid_ids: dict[int, int],
) -> torch.Tensor:

    mask = torch.zeros_like(logits, dtype=bool)
    mask[:, list(valid_ids.keys())] = 1
    masked_logits = logits.masked_fill(~mask, -float("inf"))
    return masked_logits


def regex_generate(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    regex: str,
    max_new_tokens: int = 64,
) -> str:

    if not hasattr(tokenizer, "eos_token_id"):
        raise TypeError("tokenizer does not have 'eos_token_id'")

    id_to_str = _build_id_to_str(tokenizer)
    fsm = interegular.parse_pattern(regex).to_fsm()
    state_to_valid_tokens_map = _build_state_to_valid_tokens_map(
        fsm,
        id_to_str,
        tokenizer.eos_token_id,
    )
    current_fsm_state = fsm.initial

    model_inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs["attention_mask"]

    past_key_values = None
    generated = ""

    for _ in range(max_new_tokens):
        with torch.inference_mode():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                past_key_values=past_key_values,
            )

            logits: torch.Tensor = outputs.logits
            past_key_values = outputs.past_key_values

            next_token_logits = logits[:, -1, :]

            masked_logits = _mask_logits(
                logits=next_token_logits,
                valid_ids=state_to_valid_tokens_map[current_fsm_state],
            )
            next_tokens = masked_logits.argmax(dim=-1, keepdim=True)

            next_token = next_tokens[0, 0].item()
            if next_token == tokenizer.eos_token_id:
                break

            current_fsm_state = state_to_valid_tokens_map[current_fsm_state][next_token]
            generated += id_to_str[next_token]

            if current_fsm_state in fsm.finals and len(fsm.map[current_fsm_state]) == 0:
                break

            input_ids = next_tokens
            attention_mask = torch.cat(
                (
                    attention_mask,
                    torch.ones(
                        (attention_mask.shape[0], 1),
                        dtype=attention_mask.dtype,
                        device=model.device,
                    ),
                ),
                dim=-1,
            )

    return generated
