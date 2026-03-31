from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from lisai.evaluation.cli import _resolve_evaluate_run_selector
from lisai.runs.cli import add_run_filter_arguments
from lisai.runs.scanner import default_datasets_root


def run_open_from_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    resolved = _resolve_evaluate_run_selector(args, parser=parser)
    if resolved is None:
        return 1
    dataset_name, model_subfolder, model_name = resolved
    run_dir = _resolve_run_folder(dataset_name, model_subfolder, model_name)
    if _try_open_windows_file_explorer(run_dir):
        return 0
    print(run_dir)
    return 0


def add_open_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "run",
        nargs="?",
        help=(
            "Run selector: dataset[/subfolder]/run_name, or run_name when paired with run_index. "
            "Use --run-id as an alternative selector."
        ),
    )
    parser.add_argument("run_index", nargs="?", type=int, help="Run index used with a run_name selector.")
    parser.add_argument("--run-id", help="Stable run identifier to open.")
    add_run_filter_arguments(parser, include_identity=False, include_status=False)
    return parser


def add_open_subparser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]):
    parser = subparsers.add_parser(
        "open",
        help="Open a run folder in Windows File Explorer.",
        description="Open a run folder in Windows File Explorer.",
    )
    add_open_arguments(parser)
    parser.set_defaults(handler=lambda args, p=parser: run_open_from_args(args, p))
    return parser


def build_parser(*, prog: str = "lisai open") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Open a run folder in Windows File Explorer.", prog=prog)
    add_open_arguments(parser)
    parser.set_defaults(handler=lambda args, p=parser: run_open_from_args(args, p))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return args.handler(args)


def _resolve_run_folder(dataset_name: str, model_subfolder: str, model_name: str) -> Path:
    models_root = default_datasets_root() / dataset_name / "models"
    if model_subfolder:
        models_root = models_root / Path(model_subfolder)
    return (models_root / model_name).resolve()


def _try_open_windows_file_explorer(path: Path) -> bool:
    resolved = path.resolve()
    if not resolved.exists():
        return False

    startfile = getattr(os, "startfile", None)
    if callable(startfile):
        try:
            startfile(str(resolved))
            return True
        except OSError:
            pass

    explorer_cmd = shutil.which("explorer.exe") or shutil.which("explorer")
    if explorer_cmd is None:
        return False

    windows_path = _to_windows_path(resolved)
    target = windows_path if windows_path is not None else str(resolved)
    try:
        subprocess.Popen(
            [explorer_cmd, target],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return False
    return True


def _to_windows_path(path: Path) -> str | None:
    wslpath_cmd = shutil.which("wslpath")
    if wslpath_cmd is None:
        return None
    try:
        completed = subprocess.run(
            [wslpath_cmd, "-w", str(path)],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    converted = completed.stdout.strip()
    return converted or None


__all__ = [
    "add_open_subparser",
    "build_parser",
    "main",
    "run_open_from_args",
]
