# lmdec

`lmdec` is a Python library for implementing and exploring language-model
decoding algorithms. The project currently focuses on correctness and clear
implementations, with inference performance optimizations planned as it
matures.

The package is available on [PyPI](https://pypi.org/project/lmdec/).

## Installation

`lmdec` requires Python 3.12 or newer.

```bash
pip install lmdec
```

## Current API

The initial API provides greedy generation constrained by a regular
expression. At each decoding step, `regex_generate` masks tokens that would
make the generated text violate the expression.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

from lmdec import regex_generate

model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

result = regex_generate(
    model=model,
    tokenizer=tokenizer,
    prompt="The release date is ",
    regex=r"\d{4}-\d{2}-\d{2}",
    max_new_tokens=10,
)

print(result)
```

```python
regex_generate(
    model,
    tokenizer,
    prompt: str,
    regex: str,
    max_new_tokens: int = 64,
) -> str
```

## Roadmap

Planned areas of work include:

- Structured generation beyond regular-expression constraints
- Speculative decoding
- Paged attention
- Further inference and memory optimizations

The API is still early and may change as these capabilities are developed.

## Development

Install the development dependencies and run the test suite with:

```bash
uv sync --dev
uv run pytest
```
