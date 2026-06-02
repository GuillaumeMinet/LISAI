from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

from .run_training import run_training
from lisai.config import settings

config_dir = settings.TRAINING_CONFIG_DIR
config_suffix = settings.CONFIG_SUFFIXES


def _candidate_paths(path: Path) -> tuple[Path, ...]:
    candidates = [path]
    if not path.suffix:
        candidates.extend(path.with_suffix(suffix) for suffix in config_suffix)
    return tuple(candidates)


def _first_existing_path(candidates: Iterable[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def _available_training_configs() -> list[str]:
    available: set[str] = set()
    for suffix in config_suffix:
        available.update(path.name for path in config_dir.glob(f"*{suffix}") if path.is_file())
    return sorted(available)


def _missing_config_error(config_arg: str) -> FileNotFoundError:
    available = _available_training_configs()
    lines = [f"Training config not found: {config_arg}"]
    if available:
        lines.append("Available configs:")
        lines.extend(f"  - {config_name}" for config_name in available)
    else:
        lines.append(f"No training configs were found under {config_dir}.")
    return FileNotFoundError("\n".join(lines))


def resolve_config_path(config_arg: str) -> Path:

    # first case: user gave full path
    config_path = Path(config_arg).expanduser()
    resolved = _first_existing_path(_candidate_paths(config_path))
    if resolved is not None:
        return resolved

    # second case, user gave direct config name
    if not config_path.is_absolute():
        resolved = _first_existing_path(_candidate_paths(config_dir / config_path))
        if resolved is not None:
            return resolved

    raise _missing_config_error(config_arg)


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
        help=f"Path to a YAML config file, or a config name from {config_dir} with or without .yml/.yaml.",
    )
    parser.add_argument(
        "-c",
        "--config",
        dest="config_option",
        help=f"Path to a YAML config file, or a config name from {config_dir} with or without .yml/.yaml.",
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
