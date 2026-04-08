from __future__ import annotations

import argparse
from typing import Sequence

from lisai.data.preprocess.cli import add_preprocess_subparser
from lisai.evaluation.cli import add_apply_subparser, add_evaluate_subparser
from lisai.queue.cli import add_queue_subparser
from lisai.runs.cli import add_runs_subparser
from lisai.training.cli import add_train_subparser
from lisai.training.continue_cli import add_continue_subparser
from lisai.unified_cli import add_list_subparser, add_logs_subparser, add_show_subparser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lisai", description="LISAI command line interface.")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.required = True

    add_train_subparser(subparsers)
    add_continue_subparser(subparsers)
    add_list_subparser(subparsers)
    add_show_subparser(subparsers)
    add_logs_subparser(subparsers)
    add_runs_subparser(subparsers)
    add_queue_subparser(subparsers)
    add_preprocess_subparser(subparsers)
    add_apply_subparser(subparsers)
    add_evaluate_subparser(subparsers)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return args.handler(args)
