import argparse

from lmdec.analyze import AnalyseArgs, get_analyze_parser
from lmdec.analyze import run as run_analyze


def main():
    p = argparse.ArgumentParser(prog="lmdec")
    subparsers = p.add_subparsers(dest="command", required=True)

    _ = subparsers.add_parser(
        "analyze", parents=[get_analyze_parser()], help="Run analyze"
    )

    args = p.parse_args()

    match args.command:
        case "analyze":
            run_analyze(
                AnalyseArgs(
                    model_id=args.model_id,
                    context=args.context,
                    batch_size=args.batch_size,
                    dtype=args.dtype,
                    kv_dtype=args.kv_dtype,
                    explain=args.explain,
                )
            )

        case _:
            raise ValueError(f"Unrecognized subcommand: {args.command}")
