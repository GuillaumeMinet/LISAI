from __future__ import annotations

import argparse
import re
import sys
from datetime import timedelta
from pathlib import Path
from typing import Sequence

from lisai.runs.schema import format_timestamp, utc_now
from lisai.training.cli import resolve_config_path

from .history import load_scheduling_context
from .schema import RESOURCE_CLASSES, QueueJob, ResourceClass
from .state import create_queued_job
from .storage import discover_jobs, remove_job_file
from .worker import QueueWorker


def submit_job(
    *,
    config: str,
    resource_class: ResourceClass,
    device: str,
    stdout=None,
) -> int:
    out = sys.stdout if stdout is None else stdout
    resolved_config = resolve_config_path(config)
    context = load_scheduling_context(resolved_config)

    record = create_queued_job(
        config_path=resolved_config,
        resource_class=resource_class,
        device=device,
        dataset=context.dataset,
        model_subfolder=context.model_subfolder,
        run_name=context.run_name,
        training_signature=context.training_signature,
    )
    print(f"Submitted job {record.job.job_id}", file=out)
    print(f"  config: {record.job.config}", file=out)
    print(f"  resource_class: {record.job.resource_class}", file=out)
    print(f"  device: {record.job.device}", file=out)
    return 0


def list_jobs(*, stdout=None, stderr=None) -> int:
    out = sys.stdout if stdout is None else stdout
    err = sys.stderr if stderr is None else stderr

    jobs, invalid = discover_jobs()
    if jobs:
        print(render_jobs_table([record.job for record in jobs]), file=out)
    else:
        print("No queue jobs found.", file=out)

    for item in invalid:
        print(
            f"warning: skipped invalid queue job {item.path} ({item.kind}: {item.message})",
            file=err,
        )
    return 0


def start_worker(
    *,
    poll_seconds: int | None,
    safety_margin_mb: int | None,
    once: bool = False,
    stdout=None,
    stderr=None,
) -> int:
    worker = QueueWorker(
        poll_seconds=poll_seconds,
        safety_margin_mb=safety_margin_mb,
        stdout=stdout,
        stderr=stderr,
    )
    if once:
        worker.run_once()
        return 0
    return worker.run_forever()


def clean_jobs(*, older_than: str, stdout=None) -> int:
    out = sys.stdout if stdout is None else stdout
    age_delta = parse_age_duration(older_than)
    now = utc_now()
    threshold = now - age_delta

    removed = 0
    for status in ("done", "failed"):
        records, _invalid = discover_jobs(status=status)
        for record in records:
            reference = record.job.finished_at or record.job.updated_at
            if reference < threshold:
                remove_job_file(record)
                removed += 1

    print(f"Removed {removed} queue jobs older than {older_than}.", file=out)
    return 0


def render_jobs_table(jobs: list[QueueJob]) -> str:
    if not jobs:
        return ""

    headers = [
        "job_id",
        "status",
        "config",
        "device",
        "submitted_at",
        "run_id",
    ]
    rows = [
        [
            job.job_id,
            job.status,
            _shorten_path(job.config),
            job.device,
            format_timestamp(job.submitted_at),
            "-" if job.run_id is None else job.run_id,
        ]
        for job in jobs
    ]
    widths = [max(len(header), *(len(row[idx]) for row in rows)) for idx, header in enumerate(headers)]

    lines = [
        "  ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers)),
        "  ".join("-" * widths[idx] for idx in range(len(headers))),
    ]
    lines.extend(
        "  ".join(value.ljust(widths[idx]) for idx, value in enumerate(row))
        for row in rows
    )
    return "\n".join(lines)


def parse_age_duration(value: str) -> timedelta:
    match = re.fullmatch(r"\s*(\d+)\s*([smhd])\s*", value)
    if match is None:
        raise ValueError("Duration must match <int><unit>, e.g. 7d, 12h, 30m.")
    amount = int(match.group(1))
    unit = match.group(2)
    if amount < 0:
        raise ValueError("Duration amount must be >= 0.")
    if unit == "s":
        return timedelta(seconds=amount)
    if unit == "m":
        return timedelta(minutes=amount)
    if unit == "h":
        return timedelta(hours=amount)
    return timedelta(days=amount)


def _shorten_path(path: str, *, max_chars: int = 64) -> str:
    text = str(Path(path))
    if len(text) <= max_chars:
        return text
    return f"...{text[-(max_chars - 3):]}"


def run_submit_from_args(args: argparse.Namespace) -> int:
    return submit_job(
        config=args.config,
        resource_class=args.resource_class,
        device=args.device,
    )


def run_list_from_args(args: argparse.Namespace) -> int:
    return list_jobs()


def run_worker_from_args(args: argparse.Namespace) -> int:
    return start_worker(
        poll_seconds=args.poll_seconds,
        safety_margin_mb=args.safety_margin_mb,
        once=args.once,
    )


def run_clean_from_args(args: argparse.Namespace) -> int:
    return clean_jobs(older_than=args.older_than)


def _add_queue_commands(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    submit_parser = subparsers.add_parser(
        "submit",
        help="Submit a training config to the local queue.",
        description="Submit a training config to the local queue.",
    )
    submit_parser.add_argument("--config", required=True, help="Training config path or short name.")
    submit_parser.add_argument(
        "--resource-class",
        choices=RESOURCE_CLASSES,
        default="medium",
        help="Resource class used when no historical VRAM data is available.",
    )
    submit_parser.add_argument("--device", default="cuda:0", help="Target device (default: cuda:0).")
    submit_parser.set_defaults(handler=run_submit_from_args)

    list_parser = subparsers.add_parser(
        "list",
        help="List queue jobs.",
        description="List queue jobs.",
    )
    list_parser.set_defaults(handler=run_list_from_args)

    worker_parser = subparsers.add_parser(
        "worker",
        help="Run the local queue worker loop.",
        description="Run the local queue worker loop.",
    )
    worker_parser.add_argument("--poll-seconds", type=int, default=None, help="Worker poll interval.")
    worker_parser.add_argument(
        "--safety-margin-mb",
        type=int,
        default=None,
        help="Extra free VRAM required before launch.",
    )
    worker_parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single worker cycle and exit.",
    )
    worker_parser.set_defaults(handler=run_worker_from_args)

    clean_parser = subparsers.add_parser(
        "clean",
        help="Clean old jobs from done/failed folders.",
        description="Clean old jobs from done/failed folders.",
    )
    clean_parser.add_argument(
        "--older-than",
        required=True,
        help="Age threshold like 7d, 12h, or 30m.",
    )
    clean_parser.set_defaults(handler=run_clean_from_args)


def add_queue_subparser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]):
    parser = subparsers.add_parser(
        "queue",
        help="Manage local training queue jobs.",
        description="Manage local training queue jobs.",
    )
    queue_subparsers = parser.add_subparsers(dest="queue_command")
    queue_subparsers.required = True
    _add_queue_commands(queue_subparsers)
    return parser


def build_parser(*, prog: str = "lisai queue") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage local training queue jobs.", prog=prog)
    subparsers = parser.add_subparsers(dest="queue_command")
    subparsers.required = True
    _add_queue_commands(subparsers)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return args.handler(args)


__all__ = [
    "add_queue_subparser",
    "clean_jobs",
    "list_jobs",
    "build_parser",
    "main",
    "parse_age_duration",
    "render_jobs_table",
    "start_worker",
    "submit_job",
]
