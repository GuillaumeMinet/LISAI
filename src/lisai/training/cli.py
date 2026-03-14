from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

from .run_training import run_training

EXPERIMENT_CONFIG_SUFFIXES = (".yml", ".yaml")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _candidate_paths(path: Path) -> tuple[Path, ...]:
    candidates = [path]
    if not path.suffix:
        candidates.extend(path.with_suffix(suffix) for suffix in EXPERIMENT_CONFIG_SUFFIXES)
    return tuple(candidates)


def _first_existing_path(candidates: Iterable[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def _search_roots(*, cwd: Path) -> tuple[Path, ...]:
    roots = [cwd / "configs" / "experiments"]

    repo_experiments = _repo_root() / "configs" / "experiments"
    if repo_experiments not in roots:
        roots.append(repo_experiments)

    return tuple(roots)


def _available_training_configs(search_roots: Iterable[Path]) -> list[str]:
    available: set[str] = set()
    for root in search_roots:
        if not root.is_dir():
            continue
        for suffix in EXPERIMENT_CONFIG_SUFFIXES:
            available.update(path.name for path in root.glob(f"*{suffix}") if path.is_file())
    return sorted(available)


def _missing_config_error(config_arg: str, *, search_roots: Iterable[Path]) -> FileNotFoundError:
    available = _available_training_configs(search_roots)
    lines = [f"Training config not found: {config_arg}"]
    if available:
        lines.append("Available configs:")
        lines.extend(f"  - {config_name}" for config_name in available)
    else:
        lines.append("No training configs were found under configs/experiments.")
    return FileNotFoundError("\n".join(lines))


def resolve_config_path(config_arg: str, *, cwd: Path | None = None) -> Path:
    config_path = Path(config_arg).expanduser()
    resolved = _first_existing_path(_candidate_paths(config_path))
    if resolved is not None:
        return resolved

    base_cwd = Path.cwd() if cwd is None else Path(cwd)
    search_roots = _search_roots(cwd=base_cwd)

    if not config_path.is_absolute():
        for root in search_roots:
            resolved = _first_existing_path(_candidate_paths(root / config_path))
            if resolved is not None:
                return resolved

    raise _missing_config_error(config_arg, search_roots=search_roots)


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
        help="Path to a YAML config file, or a config name from configs/experiments with or without .yml/.yaml.",
    )
    parser.add_argument(
        "-c",
        "--config",
        dest="config_option",
        help="Path to a YAML config file, or a config name from configs/experiments with or without .yml/.yaml.",
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
