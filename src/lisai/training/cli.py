from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .run_training import run_training


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_config_path(config_arg: str, *, cwd: Path | None = None) -> Path:
    config_path = Path(config_arg).expanduser()
    if config_path.exists():
        return config_path.resolve()

    search_roots = []
    base_cwd = Path.cwd() if cwd is None else Path(cwd)
    search_roots.append(base_cwd / "configs" / "experiments")

    repo_experiments = _repo_root() / "configs" / "experiments"
    if repo_experiments not in search_roots:
        search_roots.append(repo_experiments)

    for root in search_roots:
        candidate = root / config_arg
        if candidate.exists():
            return candidate.resolve()

    return config_path


def _get_config_arg(args: argparse.Namespace, parser: argparse.ArgumentParser) -> str:
    positional = getattr(args, "config", None)
    option = getattr(args, "config_option", None)

    if positional and option:
        parser.error("Provide the training config either positionally or via --config, not both.")
    if positional:
        return positional
    if option:
        return option

    parser.error("A training config is required.")
    raise AssertionError("argparse.error should raise SystemExit")


def run_from_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    config_arg = _get_config_arg(args, parser)
    run_training(resolve_config_path(config_arg))
    return 0


def add_train_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "config",
        nargs="?",
        help="Path to a YAML config file, or a config name from configs/experiments.",
    )
    parser.add_argument(
        "-c",
        "--config",
        dest="config_option",
        help="Path to a YAML config file, or a config name from configs/experiments.",
    )
    return parser


def build_parser(*, prog: str = "lisai train") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a model using a YAML config", prog=prog)
    add_train_arguments(parser)
    return parser


def add_train_subparser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]):
    parser = subparsers.add_parser(
        "train",
        help="Train a model from a YAML config.",
        description="Train a model using a YAML config",
    )
    add_train_arguments(parser)
    parser.set_defaults(handler=lambda args, p=parser: run_from_args(args, p))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return run_from_args(args, parser)
