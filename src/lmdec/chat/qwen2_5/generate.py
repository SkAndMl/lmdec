import torch
from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask, create_block_mask

from lmdec.chat.qwen2_5.model import Qwen2Model


def causal_mask(b, h, q, k):
    return q >= k


def sample(logits: Tensor, temperature: float):
    logits = logits / max(temperature, 1e-5)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs[:, -1:].argmax(dim=-1)


def prefill(
    model: Qwen2Model,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    **sampling_kwargs,
) -> torch.Tensor:
    # position_ids -> [seq_len,]
    # x            -> [b, seq_len,]
    mask = create_block_mask(
        causal_mask,
        1,
        1,
        position_ids.shape[0],
        model.max_seq_length,
        device=input_ids.device,
    )
    logits = model(
        input_ids=input_ids,
        position_ids=position_ids,
        mask=mask,
    )
    return sample(logits=logits, **sampling_kwargs)


def decode_one_token(
    model: Qwen2Model,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    block_mask: BlockMask,
    **sampling_kwargs,
) -> torch.Tensor:
    block_index = position_ids // block_mask.BLOCK_SIZE[0]
    mask = block_mask[:, :, block_index]
    mask.mask_mod = block_mask.mask_mod
    mask.seq_lengths = (1, model.max_seq_length)
    logits = model(
        input_ids=input_ids,
        position_ids=position_ids,
        mask=mask,
    )
    return sample(logits=logits, **sampling_kwargs)


def decode_n_tokens(
    model: Qwen2Model,
    cur_token: torch.Tensor,
    position_ids: torch.Tensor,
    num_new_tokens: int,
    **sampling_kwargs,
) -> list[int]:
    block_mask = create_block_mask(
        causal_mask,
        1,
        1,
        model.max_seq_length,
        model.max_seq_length,
        device=cur_token.device,
    )
    new_tokens = []

    for _ in range(num_new_tokens):
        next_token = decode_one_token(
            model, cur_token, position_ids, block_mask, **sampling_kwargs
        )
        new_tokens.append(next_token[0].item())
        position_ids += 1
        cur_token = next_token.clone()

    return new_tokens
