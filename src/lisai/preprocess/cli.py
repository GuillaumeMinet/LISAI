from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Iterable, Sequence

from lisai.config import load_yaml, settings
from lisai.infra.paths import Paths

from .reporting import ConsolePreprocessReporter, PreprocessReporter
from .run_preprocess import ExistingPreprocessOutput, PreprocessRun

config_dir = settings.PREPROCESS_CONFIG_DIR
config_suffix = settings.CONFIG_SUFFIXES


class PreprocessAbortedError(RuntimeError):
    pass

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


def _available_preprocess_configs() -> list[str]:
    available: set[str] = set()
    for suffix in config_suffix:
        available.update(path.name for path in config_dir.glob(f"*{suffix}") if path.is_file())
    return sorted(available)


def _missing_config_error(config_arg: str) -> FileNotFoundError:
    available = _available_preprocess_configs()
    lines = [f"Preprocess config not found: {config_arg}"]
    if available:
        lines.append("Available configs:")
        lines.extend(f"  - {config_name}" for config_name in available)
    else:
        lines.append(f"No preprocess configs were found under {config_dir}.")
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
        parser.error("Provide the preprocess config either positionally or via --config, not both.")
    if positional:
        return positional
    if option:
        return option

    parser.error("A preprocess config is required.")
    raise AssertionError("argparse.error should raise SystemExit")


def _confirm_overwrite(
    existing_output: ExistingPreprocessOutput,
    *,
    input_fn: Callable[[str], str],
) -> bool:
    prompt = (
        f"{existing_output.describe()}\n"
        "Overwriting will delete the current preprocess content and log. Continue? [y/N]: "
    )
    answer = input_fn(prompt).strip().lower()
    return answer in {"y", "yes"}


def run_preprocess_config(
    config_path: str | Path,
    *,
    paths: Paths | None = None,
    reporter: PreprocessReporter | None = None,
    input_fn: Callable[[str], str] = input,
    interactive: bool | None = None,
):
    cfg = load_yaml(config_path)
    runtime_paths = Paths(settings) if paths is None else paths
    active_reporter = ConsolePreprocessReporter() if reporter is None else reporter

    run = PreprocessRun.from_cfg(cfg, paths=runtime_paths)
    existing_output = run.existing_output()

    overwrite = False
    if existing_output.exists:
        is_interactive = sys.stdin.isatty() if interactive is None else interactive
        if not is_interactive:
            raise FileExistsError(existing_output.describe())
        if not _confirm_overwrite(existing_output, input_fn=input_fn):
            raise PreprocessAbortedError("Preprocess aborted. Existing preprocess output was left untouched.")
        overwrite = True

    return run.execute(overwrite=overwrite, reporter=active_reporter)


def run_from_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    config_arg = _get_config_arg(args, parser)
    try:
        run_preprocess_config(resolve_config_path(config_arg))
    except (FileExistsError, PreprocessAbortedError) as exc:
        parser.exit(status=1, message=f"{exc}\n")
    return 0


def add_preprocess_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "config",
        nargs="?",
        help="Path to a YAML config file, or a config name from configs/preprocess with or without .yml/.yaml.",
    )
    parser.add_argument(
        "-c",
        "--config",
        dest="config_option",
        help="Path to a YAML config file, or a config name from configs/preprocess with or without .yml/.yaml.",
    )
    return parser


def build_parser(*, prog: str = "lisai preprocess") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run dataset preprocessing from a YAML config", prog=prog)
    add_preprocess_arguments(parser)
    return parser


def add_preprocess_subparser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]):
    parser = subparsers.add_parser(
        "preprocess",
        help="Run dataset preprocessing from a YAML config.",
        description="Run dataset preprocessing from a YAML config",
    )
    add_preprocess_arguments(parser)
    parser.set_defaults(handler=lambda args, p=parser: run_from_args(args, p))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return run_from_args(args, parser)
