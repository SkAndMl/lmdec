from types import SimpleNamespace

import pytest
import torch

from lmdec.generate import regex_generate


class ModelInputs(dict[str, torch.Tensor]):
    def to(self, device: torch.device) -> "ModelInputs":
        return ModelInputs({name: tensor.to(device) for name, tensor in self.items()})


class CharacterTokenizer:
    eos_token_id = 3
    pad_token_id = 3
    _tokens = ("a", "b", "x", "!")

    def __len__(self) -> int:
        return len(self._tokens)

    def convert_ids_to_tokens(self, token_id: int) -> str:
        return self._tokens[token_id]

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        return "".join(tokens)

    def __call__(
        self,
        prompts: list[str],
        return_tensors: str,
        padding: bool,
        padding_side: str,
    ) -> ModelInputs:
        assert prompts == ["prompt"]
        assert return_tensors == "pt"
        assert padding is True
        assert padding_side == "left"
        return ModelInputs(
            input_ids=torch.tensor([[99]]),
            attention_mask=torch.tensor([[1]]),
        )

    def decode(self, token_ids: list[int], skip_special_tokens: bool) -> str:
        assert skip_special_tokens is True
        return "".join(
            self._tokens[token_id]
            for token_id in token_ids
            if token_id != self.eos_token_id
        )


class CacheAwareModel:
    device = torch.device("cpu")
    config = SimpleNamespace(vocab_size=4)

    def __init__(self, scores: list[list[float]]) -> None:
        self._scores = iter(scores)
        self._call_number = 0

    def __call__(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        use_cache: bool,
        past_key_values: object,
    ) -> SimpleNamespace:
        expected_input_ids = [[99]] if self._call_number == 0 else [[0]]
        assert input_ids.tolist() == expected_input_ids
        assert attention_mask.tolist() == [[1] * (self._call_number + 1)]
        assert use_cache is True
        assert past_key_values == (
            None if self._call_number == 0 else f"cache-{self._call_number}"
        )

        self._call_number += 1
        logits = torch.tensor([[next(self._scores)]])
        return SimpleNamespace(
            logits=logits,
            past_key_values=f"cache-{self._call_number}",
        )


class BatchedCharacterTokenizer(CharacterTokenizer):
    def __call__(
        self,
        prompts: list[str],
        return_tensors: str,
        padding: bool,
        padding_side: str,
    ) -> ModelInputs:
        assert return_tensors == "pt"
        assert padding is True
        assert padding_side == "left"
        batch_size = len(prompts)
        return ModelInputs(
            input_ids=torch.full((batch_size, 1), 99),
            attention_mask=torch.ones((batch_size, 1), dtype=torch.long),
        )


class BatchedModel:
    device = torch.device("cpu")
    config = SimpleNamespace(vocab_size=4)

    def __init__(self, scores: list[list[list[float]]]) -> None:
        self._scores = iter(scores)

    def __call__(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        use_cache: bool,
        past_key_values: object,
    ) -> SimpleNamespace:
        scores = torch.tensor(next(self._scores)).unsqueeze(1)
        assert scores.shape[0] == input_ids.shape[0]
        return SimpleNamespace(logits=scores, past_key_values=None)


def test_regex_generate_generates_each_prompt_independently_in_a_batch() -> None:
    model = BatchedModel(
        scores=[
            [
                [0.0, 0.0, 0.0, 10.0],
                [10.0, 0.0, 0.0, 0.0],
            ],
            [
                [10.0, 0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0, 0.0],
            ],
        ]
    )

    generated = regex_generate(
        model,
        BatchedCharacterTokenizer(),
        ["first prompt", "second prompt"],
        r"a*",
        max_new_tokens=2,
    )

    assert generated == ["", "aa"]


def test_regex_generate_keeps_finished_rows_at_eos_while_the_batch_continues() -> None:
    model = BatchedModel(
        scores=[
            [
                [0.0, 0.0, 0.0, 10.0],
                [10.0, 0.0, 0.0, 0.0],
            ],
            [
                [0.0, 10.0, 0.0, 0.0],
                [10.0, 0.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0, 10.0],
                [10.0, 0.0, 0.0, 0.0],
            ],
        ]
    )

    generated = regex_generate(
        model,
        BatchedCharacterTokenizer(),
        ["finished early", "still active"],
        r"a*(bc)?",
        max_new_tokens=3,
    )

    assert generated == ["", "aaa"]


def test_regex_generate_rejects_a_batch_when_one_row_has_no_valid_continuation() -> None:
    model = BatchedModel(
        scores=[
            [
                [10.0, 0.0, 0.0, 0.0],
                [0.0, 10.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0, 10.0],
                [10.0, 0.0, 0.0, 0.0],
            ],
        ]
    )

    with pytest.raises(RuntimeError, match="No valid continuation available"):
        regex_generate(
            model,
            BatchedCharacterTokenizer(),
            ["first prompt", "second prompt"],
            r"a|bc",
            max_new_tokens=2,
        )


def test_regex_generate_uses_only_tokens_allowed_by_the_regex() -> None:
    model = CacheAwareModel(
        scores=[
            [5.0, 4.0, 100.0, 3.0],
            [3.0, 5.0, 100.0, 4.0],
        ]
    )

    generated = regex_generate(
        model,
        CharacterTokenizer(),
        ["prompt"],
        r"ab",
        max_new_tokens=2,
    )

    assert generated == ["ab"]


def test_regex_generate_stops_at_eos_after_reaching_an_accepting_state() -> None:
    model = CacheAwareModel(
        scores=[
            [5.0, 4.0, 100.0, 3.0],
            [5.0, 4.0, 100.0, 10.0],
        ]
    )

    generated = regex_generate(
        model,
        CharacterTokenizer(),
        ["prompt"],
        r"a+",
        max_new_tokens=2,
    )

    assert generated == ["a"]


def test_regex_generate_respects_the_new_token_limit() -> None:
    model = CacheAwareModel(
        scores=[
            [10.0, 4.0, 3.0, 2.0],
            [10.0, 4.0, 3.0, 2.0],
        ]
    )

    generated = regex_generate(
        model,
        CharacterTokenizer(),
        ["prompt"],
        r"a+",
        max_new_tokens=2,
    )

    assert generated == ["aa"]


def test_regex_generate_rejects_a_tokenizer_without_an_eos_token() -> None:
    with pytest.raises(
        TypeError,
        match="tokenizer does not have 'eos_token_id'",
    ):
        regex_generate(object(), object(), ["prompt"], r"a", max_new_tokens=1)
