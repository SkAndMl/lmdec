# lmdec

`lmdec` is a Python library for implementing and exploring language-model
decoding algorithms, with a CLI for estimating model parameters and inference
memory. The project currently focuses on correctness and clear implementations,
with inference performance optimizations planned as it matures.

The package is available on [PyPI](https://pypi.org/project/lmdec/).

## Installation

`lmdec` requires Python 3.12 or newer.

```bash
pip install lmdec
```

## Analyze a model

`lmdec analyze` reads a Hugging Face model configuration without loading model
weights. It reports estimated parameter counts, weight memory, KV-cache memory,
model dimensions, and attention layout.

```bash
lmdec analyze Qwen/Qwen2.5-0.5B --context 4096 --batch_size 3
```

Choose weight and cache dtypes independently, and add `--explain` to show
derivations and assumptions:

```bash
lmdec analyze Qwen/Qwen2.5-0.5B \
  --context 4096 \
  --batch_size 3 \
  --dtype fp32 \
  --kv-dtype bf16 \
  --explain
```

| Option | Default | Meaning |
| --- | --- | --- |
| `model_id` | Required | Hugging Face model ID or local configuration directory |
| `--context` | `1` | Tokens per sequence; must be positive |
| `--batch_size` | `1` | Number of sequences; must be positive |
| `--dtype` | `bf16` | Weight storage dtype: `bf16`, `fp16`, or `fp32` |
| `--kv-dtype` | `bf16` | KV-cache storage dtype: `bf16`, `fp16`, or `fp32` |
| `--explain` | Off | Include derivations and memory assumptions |

Dedicated model families handle Qwen2/Qwen2.5 (`qwen2`), GLM-5 (`glm_moe_dsa`),
DeepSeek V3 (`deepseek_v3`), and DeepSeek V3.2 (`deepseek_v32`). Other model types
use a generic decoder estimate when their configuration exposes the required
dimensions; architecture-specific features may not be captured.

Memory estimates count raw parameter storage and theoretical cache tensors.
Every sequence is assumed to occupy the full requested context. The total
excludes activations, temporary workspaces, allocator overhead, and cache block
rounding, so it is not a prediction of peak GPU memory. Parameter estimates omit
normalization parameters and biases. MLA estimates assume compressed KV latents;
GLM-5 and DeepSeek V3.2 also include indexer keys at the chosen cache dtype.
Backends using expanded K/V or quantized indexer keys can use different memory.
Quantization scales and metadata are excluded.

You can also inspect the analysis in Python:

```python
from transformers import AutoConfig

from lmdec.model_families import DType, Workload, resolve_family
from lmdec.presentation import render_analysis

model_id = "Qwen/Qwen2.5-0.5B"
family = resolve_family(model_id, AutoConfig.from_pretrained(model_id))
analysis = family.analyze(
    Workload(context=4096, batch_size=3, dtype=DType.FP32, kv_dtype=DType.BF16)
)

print(analysis.params.total)
print(analysis.memory.total_bytes)
print(render_analysis(analysis, explain=True))
```

## Regex-constrained generation

The initial API provides batched greedy generation constrained by a regular
expression. At each decoding step, `regex_generate` masks tokens that would
make each generated text violate the expression. The same expression is
applied to every prompt in the batch.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

from lmdec import regex_generate

model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

results = regex_generate(
    model=model,
    tokenizer=tokenizer,
    prompts=[
        "The first release date is ",
        "The second release date is ",
    ],
    regex=r"\d{4}-\d{2}-\d{2}",
    max_new_tokens=10,
)

print(results)
```

```python
regex_generate(
    model,
    tokenizer,
    prompts: list[str],
    regex: str,
    max_new_tokens: int,
) -> list[str]
```

## How it works

For a detailed walkthrough of the implementation—including token masking,
finite-state machines, precomputed transitions, KV caching, batching, and the
remaining tokenizer edge cases—read [Regex-Constrained Generation: Making
Output Syntax a Decoding Rule](blogs/building-regex-constrained-decoding.md).

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
