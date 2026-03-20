from __future__ import annotations

import argparse
import sys
from datetime import datetime
from typing import Sequence

from lisai.config import settings
from lisai.runs.cli import add_run_filter_arguments
from lisai.runs.listing import (
    filter_runs,
    is_run_likely_active,
    is_run_likely_stale,
    render_runs_table,
    write_invalid_run_warnings,
)
from lisai.runs.scanner import DiscoveredRun, scan_runs

from .run_training import run_training_from_config_dict


def continue_run(
    *,
    run_id: str,
    dataset: str | None = None,
    model_subfolder: str | None = None,
    assume_yes: bool = False,
    force: bool = False,
    stdin=None,
    stdout=None,
    stderr=None,
    now: datetime | None = None,
) -> int:
    scan_result = scan_runs()
    out = sys.stdout if stdout is None else stdout
    err = sys.stderr if stderr is None else stderr
    in_stream = sys.stdin if stdin is None else stdin

    matches = filter_runs(
        scan_result.runs,
        run_id=run_id,
        dataset=dataset,
        model_subfolder=model_subfolder,
    )

    if not matches:
        print(f"No matching run found for run_id={run_id!r}.", file=err)
        print("Use 'lisai runs list' to inspect available runs.", file=err)
        write_invalid_run_warnings(scan_result.invalid, stderr=err)
        return 1

    if len(matches) > 1:
        print("Multiple matching runs found:", file=out)
        print(render_runs_table(matches), file=out)
        print("Rerun with --dataset and/or --subfolder to disambiguate.", file=err)
        write_invalid_run_warnings(scan_result.invalid, stderr=err)
        return 1

    selected_run = matches[0]
    print("Selected run:", file=out)
    print(render_runs_table([selected_run]), file=out)
    write_invalid_run_warnings(scan_result.invalid, stderr=err)

    if is_run_likely_active(selected_run, now=now):
        timeout_minutes = settings.project.run_tracking.active_heartbeat_timeout_minutes
        if not force:
            print(
                "Selected run still appears active based on a recent heartbeat. "
                f"Refusing to continue. Rerun with --force if you are sure it is not actually running "
                f"(current timeout: {timeout_minutes} minutes).",
                file=err,
            )
            return 1
        print(
            "warning: forcing continuation even though the selected run still appears active "
            f"(heartbeat timeout: {timeout_minutes} minutes).",
            file=err,
        )
    elif is_run_likely_stale(selected_run, now=now):
        print(
            "warning: selected run appears stale (running + not closed cleanly + old heartbeat). "
            "Continuing will reuse the same run directory.",
            file=err,
        )

    if not assume_yes:
        confirmed = _prompt_yes_no(
            "Continue training this run in place? [y/N]: ",
            stdin=in_stream,
            stdout=out,
        )
        if confirmed is None:
            print("Confirmation required. Rerun with --yes to continue non-interactively.", file=err)
            return 1
        if not confirmed:
            print("Continue cancelled.", file=err)
            return 1

    run_training_from_config_dict(_build_continue_training_config(selected_run))
    return 0


def _build_continue_training_config(run: DiscoveredRun) -> dict:
    return {
        "experiment": {"mode": "continue_training"},
        "load_model": {
            "canonical_load": True,
            "dataset_name": run.dataset,
            "subfolder": run.model_subfolder,
            "exp_name": run.metadata.run_id,
            "load_method": "state_dict",
            "best_or_last": "last",
        },
    }


def _prompt_yes_no(prompt: str, *, stdin, stdout) -> bool | None:
    is_tty = getattr(stdin, "isatty", None)
    if not callable(is_tty) or not is_tty():
        return None

    print(prompt, end="", file=stdout, flush=True)
    answer = stdin.readline()
    if answer == "":
        return None
    return answer.strip().lower() in {"y", "yes"}


def run_from_args(args: argparse.Namespace) -> int:
    return continue_run(
        run_id=args.run_id,
        dataset=args.dataset,
        model_subfolder=args.model_subfolder,
        assume_yes=args.yes,
        force=args.force,
    )


def add_continue_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("run_id", help="Run identifier to continue in place.")
    add_run_filter_arguments(parser, include_status=False)
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Skip the interactive confirmation prompt.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow continuation even if the selected run still appears active from a recent heartbeat.",
    )
    return parser


def add_continue_subparser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]):
    parser = subparsers.add_parser(
        "continue",
        help="Continue an existing training run without changing its config.",
        description="Continue an existing training run without changing its config.",
    )
    add_continue_arguments(parser)
    parser.set_defaults(handler=run_from_args)
    return parser


def build_parser(*, prog: str = "lisai continue") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Continue an existing training run without changing its config.",
        prog=prog,
    )
    add_continue_arguments(parser)
    parser.set_defaults(handler=run_from_args)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return args.handler(args)


__all__ = [
    "add_continue_subparser",
    "build_parser",
    "continue_run",
    "main",
]