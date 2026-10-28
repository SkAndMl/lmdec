from argparse import ArgumentParser

from lmdec.chat.engine import chat

CHAT_SUPPORTED_MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
]


def get_chat_parser() -> ArgumentParser:
    parser = ArgumentParser(add_help=False)
    parser.add_argument("--model_id", type=str, required=True)
    return parser


__all__ = [
    "CHAT_SUPPORTED_MODELS",
    "chat",
    "get_chat_parser",
]
