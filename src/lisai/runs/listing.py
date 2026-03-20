from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import datetime, timedelta, timezone
import sys

from lisai.config import settings

from .scanner import DiscoveredRun, InvalidRunMetadata
from .schema import format_timestamp, utc_now


def active_heartbeat_timeout() -> timedelta:
    minutes = settings.project.run_tracking.active_heartbeat_timeout_minutes
    return timedelta(minutes=minutes)


def filter_runs(
    runs: Iterable[DiscoveredRun],
    *,
    run_id: str | None = None,
    dataset: str | None = None,
    model_subfolder: str | None = None,
    status: str | None = None,
) -> list[DiscoveredRun]:
    return [
        run
        for run in runs
        if (run_id is None or run.metadata.run_id == run_id)
        and (dataset is None or run.dataset == dataset)
        and (model_subfolder is None or run.model_subfolder == model_subfolder)
        and (status is None or run.metadata.status == status)
    ]


def is_run_heartbeat_fresh(run: DiscoveredRun, *, now: datetime | None = None) -> bool:
    reference = utc_now() if now is None else now
    if reference.tzinfo is None or reference.utcoffset() is None:
        raise ValueError("Reference time must be timezone-aware.")
    reference = reference.astimezone(timezone.utc)
    if run.metadata.last_heartbeat_at >= reference:
        return True
    return (reference - run.metadata.last_heartbeat_at) <= active_heartbeat_timeout()


def is_run_likely_active(run: DiscoveredRun, *, now: datetime | None = None) -> bool:
    return (
        run.metadata.status == "running"
        and not run.metadata.closed_cleanly
        and is_run_heartbeat_fresh(run, now=now)
    )


def is_run_likely_stale(run: DiscoveredRun, *, now: datetime | None = None) -> bool:
    return (
        run.metadata.status == "running"
        and not run.metadata.closed_cleanly
        and not is_run_heartbeat_fresh(run, now=now)
    )


def render_runs_table(runs: Sequence[DiscoveredRun]) -> str:
    if not runs:
        return ""

    headers = [
        "dataset",
        "model_subfolder",
        "run_id",
        "status",
        "closed_cleanly",
        "epoch",
        "last_seen",
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


def write_invalid_run_warnings(
    invalid_runs: Iterable[InvalidRunMetadata],
    *,
    stderr=None,
) -> None:
    err = sys.stderr if stderr is None else stderr
    for invalid in invalid_runs:
        print(
            f"warning: skipped invalid run metadata {invalid.metadata_path} "
            f"({invalid.kind}: {invalid.message})",
            file=err,
        )


def _format_epoch(run: DiscoveredRun) -> str:
    if run.metadata.last_epoch is None and run.metadata.max_epoch is None:
        return "-"
    last_epoch = "-" if run.metadata.last_epoch is None else str(run.metadata.last_epoch)
    max_epoch = "-" if run.metadata.max_epoch is None else str(run.metadata.max_epoch)
    return f"{last_epoch}/{max_epoch}"


__all__ = [
    "active_heartbeat_timeout",
    "filter_runs",
    "is_run_heartbeat_fresh",
    "is_run_likely_active",
    "is_run_likely_stale",
    "render_runs_table",
    "write_invalid_run_warnings",
]