from __future__ import annotations

import argparse
from typing import Sequence

from lisai.evaluation.cli import add_apply_subparser, add_evaluate_subparser
from lisai.training.cli import add_train_subparser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lisai", description="LISAI command line interface.")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.required = True

    add_train_subparser(subparsers)
    add_apply_subparser(subparsers)
    add_evaluate_subparser(subparsers)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return args.handler(args)
