import re

import interegular
import torch
from interegular import FSM
from transformers import AutoModelForCausalLM, AutoTokenizer


def _build_id_to_str(tokenizer: AutoTokenizer, vocab_size: int) -> dict[int, str]:
    id_to_str: dict[int, str] = {}
    for token_id in range(min(len(tokenizer), vocab_size)):
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


def _get_allowed_and_transition(
    fsm: FSM,
    id_to_str: dict[int, str],
    vocab_size: int,
    eos_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:

    states = sorted(fsm.states)

    num_states = len(states)
    allowed = torch.zeros(size=(num_states, vocab_size), dtype=bool)
    transition = -1 * torch.ones(size=(num_states, vocab_size), dtype=torch.long)

    for state in states:
        valid_ids = []
        next_states = []
        for token_id, token in id_to_str.items():
            if (next_state := _walk_fsm(state, fsm, token)) is not None:
                valid_ids.append(token_id)
                next_states.append(next_state)

        allowed[state, valid_ids] = 1
        transition[state, valid_ids] = torch.tensor(next_states, dtype=torch.long)

        if state in fsm.finals:
            allowed[state, eos_token_id] = 1
            transition[state, eos_token_id] = state

    return allowed, transition


def regex_generate(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    regex: str,
    max_new_tokens: int,
) -> list[str]:

    if not hasattr(tokenizer, "eos_token_id"):
        raise TypeError("tokenizer does not have 'eos_token_id'")

    if tokenizer.eos_token_id is None:
        raise ValueError("'eos_token_id' has None value")

    if not hasattr(tokenizer, "pad_token_id"):
        raise TypeError("tokenizer does not have 'pad_token_id'")

    if tokenizer.pad_token_id is None:
        raise ValueError("'pad_token_id' has None value")

    batch_size = len(prompts)
    finished = torch.zeros((batch_size,), dtype=torch.bool, device=model.device)

    vocab_size = model.config.vocab_size

    id_to_str = _build_id_to_str(tokenizer, vocab_size)
    fsm = interegular.parse_pattern(regex).to_fsm()

    allowed, transition = _get_allowed_and_transition(
        fsm=fsm,
        id_to_str=id_to_str,
        vocab_size=vocab_size,
        eos_token_id=tokenizer.eos_token_id,
    )

    allowed = allowed.to(model.device)
    transition = transition.to(model.device)

    model_inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        padding_side="left",
    ).to(model.device)

    current_fsm_states = fsm.initial * torch.ones(
        size=(batch_size,),
        dtype=torch.long,
        device=model.device,
    )
    terminal_fsm_states = torch.tensor(
        [state for state in fsm.finals if len(fsm.map[state]) == 0],
        dtype=torch.long,
        device=model.device,
    )

    past_key_values = None
    tokens_produced = torch.empty(
        size=(batch_size, 0),
        dtype=torch.long,
        device=model.device,
    )
    generated_length = torch.zeros(
        size=(batch_size,),
        dtype=torch.long,
        device=model.device,
    )

    input_ids: torch.Tensor = model_inputs["input_ids"]
    attention_mask: torch.Tensor = model_inputs["attention_mask"]

    for _ in range(max_new_tokens):
        with torch.inference_mode():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                past_key_values=past_key_values,
            )

        logits: torch.Tensor = outputs.logits

        next_token_logits = logits[:, -1, :]
        mask = allowed[current_fsm_states]
        mask[finished] = False
        mask[finished, tokenizer.eos_token_id] = True

        has_valid_token = mask.any(dim=-1)
        if (~has_valid_token).any().item():
            raise RuntimeError("No valid continuation available")

        masked_logits = next_token_logits.masked_fill(~mask, -float("inf"))

        next_tokens = masked_logits.argmax(dim=-1, keepdim=True)
        finished |= next_tokens.squeeze() == tokenizer.eos_token_id

        if torch.all(finished):
            break

        tokens_produced = torch.cat((tokens_produced, next_tokens), dim=-1)
        generated_length = torch.where(finished, generated_length, generated_length + 1)

        current_fsm_states = transition[current_fsm_states, next_tokens.squeeze()]

        if terminal_fsm_states.numel() > 0 and torch.all(
            torch.isin(current_fsm_states, terminal_fsm_states)
        ):
            break

        # next iter state prep
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
        past_key_values: torch.Tensor = outputs.past_key_values

    generations = [
        tokenizer.decode(
            tokens_produced[i].tolist()[: generated_length[i].item()],
            skip_special_tokens=True,
        )
        for i in range(batch_size)
    ]

    regex_pattern = re.compile(regex)
    for i, g in enumerate(generations):
        if regex_pattern.fullmatch(g) is None:
            raise RuntimeError(
                f"Generated at index {i}: {g}, does not match the pattern provided. Hint: Please try increasing the token length"
            )

    return generations
