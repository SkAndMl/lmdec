from argparse import ArgumentParser
from dataclasses import dataclass

from transformers import AutoConfig

from lmdec.model_families.base import DType
from lmdec.model_families.registry import resolve_family
from lmdec.presentation import render_analysis


@dataclass(frozen=True)
class AnalyseArgs:
    model_id: str
    context: int
    batch_size: int
    dtype: DType
    kv_dtype: DType
    explain: bool

    def __post_init__(self):
        if self.context <= 0:
            raise ValueError("context cannot be negative")
        if self.batch_size <= 0:
            raise ValueError("batch_size cannot be negative")


def get_analyze_parser() -> ArgumentParser:
    parser = ArgumentParser(add_help=False)
    parser.add_argument("model_id", type=str)
    parser.add_argument("--context", type=int, required=False, default=1)
    parser.add_argument("--batch_size", type=int, required=False, default=1)
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["fp16", "bf16", "fp32"],
        required=False,
        default="bf16",
    )
    parser.add_argument(
        "--kv-dtype",
        type=str,
        choices=["fp16", "bf16", "fp32"],
        required=False,
        default="bf16",
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        required=False,
        default=False,
    )

    return parser


def run(args: AnalyseArgs) -> None:
    config = AutoConfig.from_pretrained(args.model_id)
    family = resolve_family(args.model_id, config)
    analysis = family.analyze(
        context=args.context,
        batch_size=args.batch_size,
        dtype=args.dtype,
        kv_dtype=args.kv_dtype,
        explain=args.explain,
    )
    print(render_analysis(analysis))
