from argparse import ArgumentParser
from dataclasses import dataclass

from transformers import AutoConfig

from lmdec.model_families.base import DType, Workload
from lmdec.model_families.registry import resolve_family
from lmdec.presentation import render_analysis


@dataclass(frozen=True)
class AnalyseArgs:
    model_id: str
    workload: Workload
    explain: bool = False


def get_analyze_parser() -> ArgumentParser:
    dtype_choices = [dtype.value for dtype in DType]

    parser = ArgumentParser(add_help=False)
    parser.add_argument("model_id", type=str)
    parser.add_argument("--context", type=int, required=False, default=1)
    parser.add_argument("--batch_size", type=int, required=False, default=1)
    parser.add_argument(
        "--dtype",
        type=str,
        choices=dtype_choices,
        required=False,
        default="bf16",
    )
    parser.add_argument(
        "--kv-dtype",
        type=str,
        choices=dtype_choices,
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
    analysis = family.analyze(args.workload)
    print(render_analysis(analysis, explain=args.explain))
