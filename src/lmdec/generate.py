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
    valid_ids: list[dict[int, int]],
) -> torch.Tensor:

    mask = torch.zeros_like(logits, dtype=bool)
    for i in range(len(valid_ids)):
        mask[i, list(valid_ids[i].keys())] = 1
    masked_logits = logits.masked_fill(~mask, -float("inf"))
    return masked_logits


def regex_generate(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    regex: str,
    max_new_tokens: int,
) -> list[str]:

    if not hasattr(tokenizer, "eos_token_id"):
        raise TypeError("tokenizer does not have 'eos_token_id'")

    if not hasattr(tokenizer, "pad_token_id"):
        raise TypeError("tokenizer does not have 'pad_token_id'")

    batch_size = len(prompts)
    finished = torch.zeros((batch_size,), dtype=torch.bool, device=model.device)

    id_to_str = _build_id_to_str(tokenizer)
    fsm = interegular.parse_pattern(regex).to_fsm()
    state_to_valid_tokens_map = _build_state_to_valid_tokens_map(
        fsm=fsm,
        id_to_str=id_to_str,
        eos_token_id=tokenizer.eos_token_id,
    )

    model_inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        padding_side="left",
    ).to(model.device)

    input_ids: torch.Tensor = model_inputs["input_ids"]
    attention_mask: torch.Tensor = model_inputs["attention_mask"]

    current_fsm_states = [fsm.initial for _ in range(batch_size)]
    generated = ["" for _ in range(batch_size)]

    past_key_values = None

    for _ in range(max_new_tokens):
        with torch.inference_mode():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                past_key_values=past_key_values,
            )

        logits: torch.Tensor = outputs.logits
        past_key_values: torch.Tensor = outputs.past_key_values

        next_token_logits = logits[:, -1, :]
        masked_logits = _mask_logits(
            logits=next_token_logits,
            valid_ids=[
                state_to_valid_tokens_map[_cur_state]
                for _cur_state in current_fsm_states
            ],
        )

        next_tokens = masked_logits.argmax(dim=-1, keepdim=True)
        finished |= next_tokens.squeeze() == tokenizer.eos_token_id

        if torch.all(finished):
            break

        for i in range(batch_size):
            next_token = next_tokens[i, 0].item()
            current_fsm_states[i] = state_to_valid_tokens_map[current_fsm_states[i]][
                next_token
            ]
            if next_token != tokenizer.eos_token_id:
                generated[i] += id_to_str[next_token]

        if all(
            _cur_state in fsm.finals and len(fsm.map[_cur_state]) == 0
            for _cur_state in current_fsm_states
        ):
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
