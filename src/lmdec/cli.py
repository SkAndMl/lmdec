import argparse

from .analyze import analyze


def main():
    p = argparse.ArgumentParser(prog="lmdec")
    subparsers = p.add_subparsers(dest="command", required=True)

    p_analyze = subparsers.add_parser("analyze")
    p_analyze.add_argument("model_id", type=str)

    args = p.parse_args()
    if args.command == "analyze":
        analyze(args.model_id)
