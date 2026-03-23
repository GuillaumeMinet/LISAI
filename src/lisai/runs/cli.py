from __future__ import annotations

import argparse
import sys
from typing import Sequence

from .listing import (
    filter_runs,
    has_path_inconsistencies,
    render_runs_table,
    write_invalid_run_warnings,
)
from .scanner import scan_runs
from .schema import RUN_STATUSES


def list_runs(
    *,
    run_id: str | None = None,
    run_name: str | None = None,
    run_index: int | None = None,
    dataset: str | None = None,
    model_subfolder: str | None = None,
    status: str | None = None,
    full: bool = False,
    stdout=None,
    stderr=None,
) -> int:
    scan_result = scan_runs()
    out = sys.stdout if stdout is None else stdout
    err = sys.stderr if stderr is None else stderr

    filtered_runs = filter_runs(
        scan_result.runs,
        run_id=run_id,
        run_name=run_name,
        run_index=run_index,
        dataset=dataset,
        model_subfolder=model_subfolder,
        status=status,
    )

    if filtered_runs:
        print(render_runs_table(filtered_runs, full=full), file=out)
        if has_path_inconsistencies(filtered_runs):
            print(
                "Some listed runs have inconsistent path metadata (likely moved/renamed folders).",
                file=out,
            )
    else:
        print("No runs found.", file=out)

    write_invalid_run_warnings(scan_result.invalid, stderr=err)
    return 0


def run_list_from_args(args: argparse.Namespace) -> int:
    return list_runs(
        run_id=args.run_id,
        run_name=args.run_name,
        run_index=args.run_index,
        dataset=args.dataset,
        model_subfolder=args.model_subfolder,
        status=args.status,
        full=args.full,
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
    add_run_filter_arguments(list_parser, include_status=True)
    list_parser.add_argument(
        "--full",
        action="store_true",
        help="Include extra metadata columns (path_consistent, closed_cleanly, last_seen).",
    )
    list_parser.set_defaults(handler=run_list_from_args)
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
    add_run_filter_arguments(list_parser, include_status=True)
    list_parser.add_argument(
        "--full",
        action="store_true",
        help="Include extra metadata columns (path_consistent, closed_cleanly, last_seen).",
    )
    list_parser.set_defaults(handler=run_list_from_args)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return args.handler(args)


__all__ = ["add_run_filter_arguments", "add_runs_subparser", "build_parser", "list_runs", "main"]
