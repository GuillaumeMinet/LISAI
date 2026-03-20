from __future__ import annotations

import argparse
import sys
from typing import Sequence

from .scanner import DiscoveredRun, scan_runs
from .schema import RUN_STATUSES, format_timestamp


def list_runs(
    *,
    dataset: str | None = None,
    model_subfolder: str | None = None,
    status: str | None = None,
    stdout=None,
    stderr=None,
) -> int:
    scan_result = scan_runs()
    out = sys.stdout if stdout is None else stdout
    err = sys.stderr if stderr is None else stderr

    filtered_runs = [
        run
        for run in scan_result.runs
        if (dataset is None or run.dataset == dataset)
        and (model_subfolder is None or run.model_subfolder == model_subfolder)
        and (status is None or run.metadata.status == status)
    ]

    if filtered_runs:
        print(_render_table(filtered_runs), file=out)
    else:
        print("No runs found.", file=out)

    for invalid in scan_result.invalid:
        print(
            f"warning: skipped invalid run metadata {invalid.metadata_path} "
            f"({invalid.kind}: {invalid.message})",
            file=err,
        )

    return 0


def run_list_from_args(args: argparse.Namespace) -> int:
    return list_runs(dataset=args.dataset, model_subfolder=args.model_subfolder, status=args.status)


def add_list_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--dataset", help="Filter runs by dataset name.")
    parser.add_argument(
        "--model-subfolder",
        "--models-subfolder",
        "--subfolder",
        dest="model_subfolder",
        help="Filter runs by training model_subfolder.",
    )
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
    add_list_arguments(list_parser)
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
    add_list_arguments(list_parser)
    list_parser.set_defaults(handler=run_list_from_args)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return args.handler(args)


def _render_table(runs: list[DiscoveredRun]) -> str:
    headers = [
        "dataset",
        "model_subfolder",
        "run_id",
        "status",
        "closed_cleanly",
        "epoch",
        "last_seen",
        "path",
    ]
    rows = [
        [
            run.dataset,
            run.model_subfolder,
            run.metadata.run_id,
            run.metadata.status,
            str(run.metadata.closed_cleanly).lower(),
            _format_epoch(run),
            format_timestamp(run.last_seen),
            run.path,
        ]
        for run in runs
    ]

    widths = [
        max(len(header), *(len(row[idx]) for row in rows))
        for idx, header in enumerate(headers)
    ]

    lines = [
        "  ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers)),
        "  ".join("-" * widths[idx] for idx in range(len(headers))),
    ]
    lines.extend(
        "  ".join(value.ljust(widths[idx]) for idx, value in enumerate(row))
        for row in rows
    )
    return "\n".join(lines)


def _format_epoch(run: DiscoveredRun) -> str:
    if run.metadata.last_epoch is None and run.metadata.max_epoch is None:
        return "-"
    last_epoch = "-" if run.metadata.last_epoch is None else str(run.metadata.last_epoch)
    max_epoch = "-" if run.metadata.max_epoch is None else str(run.metadata.max_epoch)
    return f"{last_epoch}/{max_epoch}"


__all__ = ["add_runs_subparser", "build_parser", "list_runs", "main"]
