from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Sequence

from .listing import (
    filter_runs,
    has_path_inconsistencies,
    render_runs_table,
    write_invalid_run_warnings,
)
from .plotting import show_loss_plot_for_run
from .scanner import DiscoveredRun, InvalidRunMetadata, scan_runs
from .schema import RUN_STATUSES
from .selection import resolve_ambiguous_run_matches

_LIVE_INTERVAL_MIN_SECONDS = 1.0


def list_runs(
    *,
    run_id: str | None = None,
    run_name: str | None = None,
    run_index: int | None = None,
    dataset: str | None = None,
    model_subfolder: str | None = None,
    status: str | None = None,
    full: bool = False,
    live: bool = False,
    interval_seconds: float = 2.0,
    stdout=None,
    stderr=None,
) -> int:
    out = sys.stdout if stdout is None else stdout
    err = sys.stderr if stderr is None else stderr
    if not math.isfinite(interval_seconds):
        raise ValueError("interval_seconds must be finite.")

    resolved_interval_seconds = interval_seconds
    interval_warning: str | None = None
    if live and interval_seconds < _LIVE_INTERVAL_MIN_SECONDS:
        resolved_interval_seconds = _LIVE_INTERVAL_MIN_SECONDS
        interval_warning = (
            f"warning: --interval {interval_seconds:g}s is below the minimum "
            f"{_LIVE_INTERVAL_MIN_SECONDS:g}s; using {_LIVE_INTERVAL_MIN_SECONDS:g}s."
        )

    if live and not _is_interactive(out):
        print(
            "warning: --live requires interactive terminal output; showing a single snapshot instead.",
            file=err,
        )
        live = False

    if live:
        emitted_invalid_keys: set[tuple[str, str, str]] = set()
        try:
            while True:
                _render_runs_snapshot(
                    run_id=run_id,
                    run_name=run_name,
                    run_index=run_index,
                    dataset=dataset,
                    model_subfolder=model_subfolder,
                    status=status,
                    full=full,
                    stdout=out,
                    stderr=err,
                    live=True,
                    refresh_interval_seconds=resolved_interval_seconds,
                    emitted_invalid_keys=emitted_invalid_keys,
                    top_notice=interval_warning,
                )
                time.sleep(resolved_interval_seconds)
        except KeyboardInterrupt:
            # Keep shell prompt on a clean line after Ctrl+C in live mode.
            print(file=out, flush=True)
            return 0

    _render_runs_snapshot(
        run_id=run_id,
        run_name=run_name,
        run_index=run_index,
        dataset=dataset,
        model_subfolder=model_subfolder,
        status=status,
        full=full,
        stdout=out,
        stderr=err,
        live=False,
        refresh_interval_seconds=resolved_interval_seconds,
        emitted_invalid_keys=None,
        top_notice=interval_warning,
    )
    return 0


def _render_runs_snapshot(
    *,
    run_id: str | None,
    run_name: str | None,
    run_index: int | None,
    dataset: str | None,
    model_subfolder: str | None,
    status: str | None,
    full: bool,
    stdout,
    stderr,
    live: bool,
    refresh_interval_seconds: float,
    emitted_invalid_keys: set[tuple[str, str, str]] | None,
    top_notice: str | None,
) -> None:
    scan_result = scan_runs()
    filtered_runs = filter_runs(
        scan_result.runs,
        run_id=run_id,
        run_name=run_name,
        run_index=run_index,
        dataset=dataset,
        model_subfolder=model_subfolder,
        status=status,
    )

    snapshot_lines: list[str] = []
    if top_notice is not None:
        snapshot_lines.append(top_notice)
    snapshot_lines.append(
        _format_listing_title(
            run_id=run_id,
            run_name=run_name,
            run_index=run_index,
            dataset=dataset,
            model_subfolder=model_subfolder,
            status=status,
            live=live,
            refresh_interval_seconds=refresh_interval_seconds,
        )
    )

    body = "No runs found."
    if filtered_runs:
        body = render_runs_table(filtered_runs, full=full)
        if has_path_inconsistencies(filtered_runs):
            body = "\n".join(
                [
                    body,
                    "",
                    "Some listed runs have inconsistent path metadata (likely moved/renamed folders).",
                ]
            )

    snapshot_lines.append(body)
    snapshot = "\n".join(snapshot_lines)

    if live:
        _print_live_snapshot(snapshot, stdout=stdout)
    else:
        print(snapshot, file=stdout)

    if emitted_invalid_keys is None:
        write_invalid_run_warnings(scan_result.invalid, stderr=stderr)
        return

    new_invalid = _filter_new_invalid_warnings(scan_result.invalid, emitted_invalid_keys)
    if new_invalid:
        write_invalid_run_warnings(new_invalid, stderr=stderr)


def _format_listing_title(
    *,
    run_id: str | None,
    run_name: str | None,
    run_index: int | None,
    dataset: str | None,
    model_subfolder: str | None,
    status: str | None,
    live: bool,
    refresh_interval_seconds: float,
) -> str:
    filter_parts: list[str] = []
    if dataset:
        filter_parts.append(f"Dataset: '{dataset}'")
    if model_subfolder:
        filter_parts.append(f"Subfolder: '{model_subfolder}'")
    if status:
        filter_parts.append(f"Status: '{status}'")
    if run_name:
        filter_parts.append(f"run_name='{run_name}'")
    if run_index is not None:
        filter_parts.append(f"run_index={run_index}")
    if run_id:
        filter_parts.append(f"run_id={run_id}")

    title = "LISAI runs listing"
    if filter_parts:
        title = f"{title} - {' | '.join(filter_parts)}\n"
    if live:
        title = f"{title}LIVE MODE ({refresh_interval_seconds:g}s refresh) - Ctrl+C to stop live\n"
    return title


def _print_live_snapshot(snapshot: str, *, stdout) -> None:
    # ANSI: move cursor to top-left and clear the rest of the screen.
    stdout.write("\x1b[H\x1b[J")
    stdout.write(snapshot)
    stdout.write("\n")
    flush = getattr(stdout, "flush", None)
    if callable(flush):
        flush()


def _filter_new_invalid_warnings(
    invalid_runs: Iterable[InvalidRunMetadata],
    seen_keys: set[tuple[str, str, str]],
) -> list[InvalidRunMetadata]:
    new_entries: list[InvalidRunMetadata] = []
    for invalid in invalid_runs:
        key = (str(invalid.metadata_path), invalid.kind, invalid.message)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        new_entries.append(invalid)
    return new_entries


def _is_interactive(stream: Any) -> bool:
    is_tty = getattr(stream, "isatty", None)
    return callable(is_tty) and bool(is_tty())


def _seconds_value(value: str) -> float:
    try:
        seconds = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid seconds value: {value!r}.") from exc
    if not math.isfinite(seconds):
        raise argparse.ArgumentTypeError("Seconds value must be finite.")
    return seconds


def run_list_from_args(args: argparse.Namespace) -> int:
    return list_runs(
        run_id=args.run_id,
        run_name=args.run_name,
        run_index=args.run_index,
        dataset=args.dataset,
        model_subfolder=args.model_subfolder,
        status=args.status,
        full=args.full,
        live=args.live,
        interval_seconds=args.interval,
    )


def _parse_run_ref_selector(run_ref: str) -> tuple[str, str, str]:
    parts = [part for part in run_ref.replace("\\", "/").split("/") if part]
    if len(parts) < 2:
        raise ValueError(
            "Run reference must be 'dataset/exp_name' or 'dataset/subfolder/exp_name'."
        )
    dataset_name = parts[0]
    run_dir_name = parts[-1]
    model_subfolder = "/".join(parts[1:-1])
    return dataset_name, model_subfolder, run_dir_name


def _resolve_single_run_selector(
    args: argparse.Namespace,
    *,
    parser: argparse.ArgumentParser,
) -> DiscoveredRun | None:
    out = sys.stdout
    err = sys.stderr

    run = args.run
    run_index = args.run_index
    run_id = args.run_id
    dataset = args.dataset
    model_subfolder = args.model_subfolder
    run_dir_name: str | None = None

    if run_id is not None and (run is not None or run_index is not None):
        print("Use either <run_name> <run_index> or --run-id, not both.", file=err)
        return None

    if run_id is None:
        if run is None:
            print(
                "Missing run selector. Use dataset[/subfolder]/run_name, <run_name> <run_index>, or --run-id <run_id>.",
                file=err,
            )
            return None

        normalized = run.replace("\\", "/")
        has_run_ref_separator = "/" in normalized
        if run_index is None and has_run_ref_separator:
            if dataset is not None or model_subfolder is not None:
                print(
                    "--dataset/--subfolder can only be used with <run_name> <run_index> or --run-id selectors.",
                    file=err,
                )
                return None
            try:
                dataset, model_subfolder, run_dir_name = _parse_run_ref_selector(run)
            except ValueError as exc:
                parser.error(str(exc))
                raise AssertionError("argparse.error should raise SystemExit")

        if run_dir_name is None:
            if run_index is None:
                print(
                    "Missing run_index for run_name selector. Use <run_name> <run_index>, "
                    "or pass dataset[/subfolder]/run_name.",
                    file=err,
                )
                return None
            if run_index < 0:
                print("run_index must be >= 0.", file=err)
                return None
            if has_run_ref_separator:
                print(
                    "run_index cannot be combined with dataset[/subfolder]/run_name selectors.",
                    file=err,
                )
                return None

    scan_result = scan_runs()
    if run_dir_name is None:
        matches = filter_runs(
            scan_result.runs,
            run_id=run_id,
            run_name=run if run_id is None else None,
            run_index=run_index if run_id is None else None,
            dataset=dataset,
            model_subfolder=model_subfolder,
        )
    else:
        matches = [
            candidate
            for candidate in scan_result.runs
            if candidate.dataset == dataset
            and candidate.model_subfolder == model_subfolder
            and candidate.run_dir.name == run_dir_name
        ]

    if not matches:
        if run_id is not None:
            selector_desc = f"run_id={run_id!r}"
        elif run_dir_name is not None:
            selector_desc = f"run={run!r}"
        else:
            selector_desc = f"run_name={run!r}, run_index={run_index}"
        print(f"No matching run found for {selector_desc}.", file=err)
        print("Use 'lisai runs list' to inspect available runs.", file=err)
        write_invalid_run_warnings(scan_result.invalid, stderr=err)
        return None

    selected = resolve_ambiguous_run_matches(
        matches,
        stdin=sys.stdin,
        stdout=out,
        stderr=err,
        rerun_hint="Rerun with --dataset/--subfolder or with --run-id to disambiguate.",
    )
    if selected is None:
        write_invalid_run_warnings(scan_result.invalid, stderr=err)
        return None

    print("Selected run:", file=out)
    print(render_runs_table([selected]), file=out)
    write_invalid_run_warnings(scan_result.invalid, stderr=err)
    return selected


def _try_open_path(path: Path) -> bool:
    resolved = path.resolve()

    startfile = getattr(os, "startfile", None)
    if callable(startfile):
        try:
            startfile(str(resolved))
            return True
        except OSError:
            pass

    commands: list[list[str]] = []
    explorer = shutil.which("explorer.exe") or shutil.which("explorer")
    if explorer is not None:
        target = _to_windows_path(resolved)
        commands.append([explorer, target if target is not None else str(resolved)])

    xdg_open = shutil.which("xdg-open")
    if xdg_open is not None:
        commands.append([xdg_open, str(resolved)])

    for command in commands:
        try:
            subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except OSError:
            continue
    return False


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


def run_open_from_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    selected = _resolve_single_run_selector(args, parser=parser)
    if selected is None:
        return 1
    if _try_open_path(selected.run_dir):
        return 0
    print(selected.run_dir)
    return 0


def run_plot_from_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    selected = _resolve_single_run_selector(args, parser=parser)
    if selected is None:
        return 1

    architecture = None
    if selected.metadata.training_signature is not None:
        architecture = selected.metadata.training_signature.architecture
    return show_loss_plot_for_run(
        run_dir=selected.run_dir,
        dataset=selected.dataset,
        model_subfolder=selected.model_subfolder,
        architecture=architecture,
        stderr=sys.stderr,
        open_saved_plot=_try_open_path,
    )


def add_run_filter_arguments(
    parser: argparse.ArgumentParser,
    *,
    include_identity: bool = True,
    include_status: bool = False,
) -> argparse.ArgumentParser:
    if include_identity:
        parser.add_argument("--run-id", help="Filter runs by stable run_id.")
        parser.add_argument("--run-name", help="Filter runs by semantic run_name.")
        parser.add_argument("--run-index", type=int, help="Filter runs by run_index.")
    parser.add_argument("--dataset", help="Filter runs by dataset name.")
    parser.add_argument(
        "--model-subfolder",
        "--models-subfolder",
        "--subfolder",
        dest="model_subfolder",
        help="Filter runs by training model_subfolder.",
    )
    if include_status:
        parser.add_argument("--status", choices=RUN_STATUSES, help="Filter runs by persisted status.")
    return parser


def _add_runs_list_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    add_run_filter_arguments(parser, include_status=True)
    parser.add_argument(
        "--full",
        action="store_true",
        help=(
            "Include extra metadata columns "
            "(failure, path_consistent, closed_cleanly, start_time, last_seen, run_id)."
        ),
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Refresh the runs table continuously (interactive terminals only).",
    )
    parser.add_argument(
        "--interval",
        type=_seconds_value,
        default=2.0,
        metavar="SECONDS",
        help="Refresh interval for --live mode (default: 2.0).",
    )
    parser.set_defaults(handler=run_list_from_args)
    return parser


def _add_runs_plot_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "run",
        nargs="?",
        help=(
            "Run selector: dataset[/subfolder]/run_name, or run_name when paired with run_index. "
            "Use --run-id as an alternative selector."
        ),
    )
    parser.add_argument("run_index", nargs="?", type=int, help="Run index used with a run_name selector.")
    parser.add_argument("--run-id", help="Stable run identifier to plot.")
    add_run_filter_arguments(parser, include_identity=False, include_status=False)
    parser.set_defaults(handler=lambda args, p=parser: run_plot_from_args(args, p))
    return parser


def _add_runs_open_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
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
    parser.set_defaults(handler=lambda args, p=parser: run_open_from_args(args, p))
    return parser


def add_runs_subparser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]):
    parser = subparsers.add_parser(
        "runs",
        help="Inspect locally tracked training runs.",
        description="Inspect locally tracked training runs.",
    )
    runs_subparsers = parser.add_subparsers(dest="runs_command")
    runs_subparsers.required = True

    list_parser = runs_subparsers.add_parser(
        "list",
        help="List locally tracked training runs.",
        description="List locally tracked training runs.",
    )
    _add_runs_list_arguments(list_parser)

    plot_parser = runs_subparsers.add_parser(
        "plot",
        help="Plot train/val losses for a selected run.",
        description="Plot train/val losses for a selected run.",
    )
    _add_runs_plot_arguments(plot_parser)

    open_parser = runs_subparsers.add_parser(
        "open",
        help="Open a selected run folder in file explorer.",
        description="Open a selected run folder in file explorer.",
    )
    _add_runs_open_arguments(open_parser)
    return parser


def build_parser(*, prog: str = "lisai runs") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect locally tracked training runs.", prog=prog)
    subparsers = parser.add_subparsers(dest="runs_command")
    subparsers.required = True

    list_parser = subparsers.add_parser(
        "list",
        help="List locally tracked training runs.",
        description="List locally tracked training runs.",
    )
    _add_runs_list_arguments(list_parser)

    plot_parser = subparsers.add_parser(
        "plot",
        help="Plot train/val losses for a selected run.",
        description="Plot train/val losses for a selected run.",
    )
    _add_runs_plot_arguments(plot_parser)

    open_parser = subparsers.add_parser(
        "open",
        help="Open a selected run folder in file explorer.",
        description="Open a selected run folder in file explorer.",
    )
    _add_runs_open_arguments(open_parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return args.handler(args)


__all__ = ["add_run_filter_arguments", "add_runs_subparser", "build_parser", "list_runs", "main"]
