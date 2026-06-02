from __future__ import annotations

import argparse
from typing import Sequence

from lisai_runner.queue.cli import add_queue_subparser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lisai-runner", description="LISAI RUNNER command line interface.")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.required = True

    add_queue_subparser(subparsers)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return args.handler(args)
