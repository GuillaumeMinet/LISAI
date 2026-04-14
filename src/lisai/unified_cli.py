from __future__ import annotations

import argparse
import os
import sys
import time
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from lisai.infra.paths import Paths
from lisai.queue.schema import JOB_STATUSES, is_queue_selector
from lisai.queue.storage import (
    DiscoveredJob,
    discover_jobs,
    job_log_filename,
    queue_logs_dir,
)
from lisai.runs.cli import add_run_filter_arguments
from lisai.runs.listing import (
    display_run_status,
    filter_runs,
    write_invalid_run_warnings,
)
from lisai.runs.scanner import DiscoveredRun, scan_runs
from lisai.runs.schema import RUN_STATUSES, format_timestamp_local
from lisai.runs.selection import resolve_ambiguous_matches

ERROR_LINE_HINTS = (
    "error",
    "exception",
    "traceback",
    "failed",
    "fatal",
    "critical",
    "runtimeerror",
    "valueerror",
)

UNIFIED_STATUS_CHOICES = tuple(dict.fromkeys((*JOB_STATUSES, *RUN_STATUSES)))
_PATHS = Paths()


@dataclass(frozen=True)
class _UnifiedRow:
    job_ref: str
    run_id: str
    dataset: str
    model_subfolder: str
    name: str
    run_idx: str
    state: str
    status_key: str
    epoch: str
    eta: str
    loss: str
    failure: str
    sort_time: datetime


@dataclass(frozen=True)
class _ResolvedTarget:
    kind: str
    job: DiscoveredJob | None = None
    run: DiscoveredRun | None = None

    @property
    def id_text(self) -> str:
        if self.kind == "job":
            assert self.job is not None
            return self.job.job.selector or self.job.job.job_id
        assert self.run is not None
        return self.run.metadata.run_id

    @property
    def dataset(self) -> str:
        if self.kind == "job":
            assert self.job is not None
            return self.job.job.dataset or "-"
        assert self.run is not None
        return self.run.dataset

    @property
    def model_subfolder(self) -> str:
        if self.kind == "job":
            assert self.job is not None
            return self.job.job.model_subfolder or "-"
        assert self.run is not None
        return self.run.model_subfolder

    @property
    def name(self) -> str:
        if self.kind == "job":
            assert self.job is not None
            return self.job.job.run_name or Path(self.job.job.config).name
        assert self.run is not None
        return self.run.metadata.run_name

    @property
    def state(self) -> str:
        if self.kind == "job":
            assert self.job is not None
            return self.job.job.status
        assert self.run is not None
        return display_run_status(self.run)


def list_targets(
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
    out = sys.stdout if stdout is None else stdout
    err = sys.stderr if stderr is None else stderr
    include_queue, queue_status, include_runs, run_status = _split_status_filter(status)

    scan_result = scan_runs()
    write_invalid_run_warnings(scan_result.invalid, stderr=err)
    run_catalog = scan_result.runs
    run_rows = (
        filter_runs(
            run_catalog,
            run_id=run_id,
            run_name=run_name,
            run_index=run_index,
            dataset=dataset,
            model_subfolder=model_subfolder,
            status=run_status,
        )
        if include_runs
        else []
    )

    queue_rows: list[DiscoveredJob] = []
    if include_queue:
        queue_records, invalid_jobs = discover_jobs(status=queue_status)
        for item in invalid_jobs:
            print(
                f"warning: skipped invalid queue job {item.path} ({item.kind}: {item.message})",
                file=err,
            )
        queue_rows = [
            record
            for record in queue_records
            if _queue_record_matches_filters(
                record,
                run_id=run_id,
                run_name=run_name,
                run_index=run_index,
                dataset=dataset,
                model_subfolder=model_subfolder,
                status=queue_status,
            )
        ]

    if not queue_rows and not run_rows:
        print("No matching jobs or runs found.", file=out)
        return 0

    run_by_id, run_by_semantic = _index_runs(run_rows)
    metrics_cache: dict[Path, tuple[float | None, float | None]] = {}
    failure_cache: dict[Path, str | None] = {}
    consumed_run_ids: set[str] = set()
    rows: list[_UnifiedRow] = []

    for record in queue_rows:
        job = record.job
        linked_run = None
        if job.status != "queued":
            linked_run = _resolve_run_for_job(
                job,
                run_by_id=run_by_id,
                run_by_semantic=run_by_semantic,
            )
        if linked_run is None:
            rows.append(
                _build_job_only_row(
                    record,
                    metrics_cache=metrics_cache,
                )
            )
            continue

        run_status = linked_run.metadata.status
        if job.status == "running":
            consumed_run_ids.add(linked_run.metadata.run_id)
            rows.append(
                _build_merged_row(
                    record,
                    linked_run,
                    metrics_cache=metrics_cache,
                    failure_cache=failure_cache,
                )
            )
            continue

        if job.status == "done":
            if run_status == "running":
                # Keep the active run view when queue metadata says this attempt is done.
                continue
            consumed_run_ids.add(linked_run.metadata.run_id)
            rows.append(
                _build_merged_row(
                    record,
                    linked_run,
                    metrics_cache=metrics_cache,
                    failure_cache=failure_cache,
                )
            )
            continue

        if job.status in {"blocked"} and run_status == "running":
            # Prefer the run-only row when a linked run is active.
            continue

        if job.status == "failed" and run_status == "running":
            # Inconsistent pair; keep queue failure explicit and do not merge.
            rows.append(
                _build_job_only_row(
                    record,
                    metrics_cache=metrics_cache,
                )
            )
            continue

        rows.append(
            _build_job_only_row(
                record,
                metrics_cache=metrics_cache,
            )
        )

    for run in run_rows:
        if run.metadata.run_id in consumed_run_ids:
            continue
        rows.append(
            _build_run_only_row(
                run,
                metrics_cache=metrics_cache,
                failure_cache=failure_cache,
            )
        )

    rows.sort(key=_row_sort_key)

    print(_render_unified_rows(rows, full=full), file=out)
    return 0


def show_target(
    *,
    target: str,
    stdout=None,
    stderr=None,
    stdin=None,
) -> int:
    out = sys.stdout if stdout is None else stdout
    err = sys.stderr if stderr is None else stderr
    in_stream = sys.stdin if stdin is None else stdin

    resolved = _resolve_target(target, stdout=out, stderr=err, stdin=in_stream)
    if resolved is None:
        return 1

    if resolved.kind == "job":
        assert resolved.job is not None
        return _show_job_details(resolved.job, stdout=out)
    assert resolved.run is not None
    return _show_run_details(resolved.run, stdout=out)


def logs_target(
    *,
    target: str,
    lines: int = 100,
    follow: bool = False,
    stdout=None,
    stderr=None,
    stdin=None,
) -> int:
    out = sys.stdout if stdout is None else stdout
    err = sys.stderr if stderr is None else stderr
    in_stream = sys.stdin if stdin is None else stdin

    resolved = _resolve_target(target, stdout=out, stderr=err, stdin=in_stream)
    if resolved is None:
        return 1

    log_path: Path
    if resolved.kind == "job":
        assert resolved.job is not None
        log_path = _resolve_canonical_job_log_path(resolved.job.job)
    else:
        assert resolved.run is not None
        log_path = _resolve_canonical_run_log_path(resolved.run)

    if not log_path.exists():
        print(f"Log file not found: {log_path}", file=err)
        return 1

    for line in _tail_lines(log_path, limit=lines):
        print(line, file=out)

    if not follow:
        return 0
    try:
        _follow_log(log_path, stdout=out)
    except KeyboardInterrupt:
        print("", file=out)
    return 0


def add_list_subparser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]):
    parser = subparsers.add_parser(
        "list",
        help="List queue jobs and runs in one unified view.",
        description="List queue jobs and runs in one unified view.",
    )
    add_run_filter_arguments(parser, include_status=False)
    parser.add_argument(
        "--status",
        choices=UNIFIED_STATUS_CHOICES,
        help=(
            "Shared status filter. "
            "running/failed apply to jobs+runs; queued/blocked/done are queue-only; "
            "completed/stopped are run-only."
        ),
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Include run_id, latest loss, and failure summary columns.",
    )
    parser.set_defaults(handler=run_list_from_args)
    return parser


def add_show_subparser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]):
    parser = subparsers.add_parser(
        "show",
        help="Show details for a queue job or run.",
        description="Show details for a queue job or run.",
    )
    parser.add_argument(
        "target",
        help=(
            "Selector/id. Supports qNNNN, job_id/prefix, run_id/prefix, or "
            "dataset[/subfolder]/run_dir_name."
        ),
    )
    parser.set_defaults(handler=run_show_from_args)
    return parser


def add_logs_subparser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]):
    parser = subparsers.add_parser(
        "logs",
        help="Show logs for a queue job or run.",
        description="Show logs for a queue job or run.",
    )
    parser.add_argument(
        "target",
        help=(
            "Selector/id. Supports qNNNN, job_id/prefix, run_id/prefix, or "
            "dataset[/subfolder]/run_dir_name."
        ),
    )
    parser.add_argument(
        "-n",
        "--lines",
        type=int,
        default=100,
        help="Number of lines to show initially (default: 100).",
    )
    parser.add_argument(
        "-f",
        "--follow",
        action="store_true",
        help="Follow log output continuously.",
    )
    parser.set_defaults(handler=run_logs_from_args)
    return parser


def run_list_from_args(args: argparse.Namespace) -> int:
    return list_targets(
        run_id=args.run_id,
        run_name=args.run_name,
        run_index=args.run_index,
        dataset=args.dataset,
        model_subfolder=args.model_subfolder,
        status=args.status,
        full=args.full,
    )


def run_show_from_args(args: argparse.Namespace) -> int:
    return show_target(target=args.target)


def run_logs_from_args(args: argparse.Namespace) -> int:
    return logs_target(
        target=args.target,
        lines=args.lines,
        follow=args.follow,
    )


def _split_status_filter(
    status: str | None,
) -> tuple[bool, str | None, bool, str | None]:
    if status is None:
        return True, None, True, None

    in_queue = status in JOB_STATUSES
    in_runs = status in RUN_STATUSES
    queue_status = status if in_queue else None
    run_status = status if in_runs else None
    return in_queue, queue_status, in_runs, run_status


def _queue_record_matches_filters(
    record: DiscoveredJob,
    *,
    run_id: str | None,
    run_name: str | None,
    run_index: int | None,
    dataset: str | None,
    model_subfolder: str | None,
    status: str | None,
) -> bool:
    job = record.job
    if status is not None and job.status != status:
        return False
    if run_id is not None and job.run_id != run_id:
        return False
    if run_name is not None and job.run_name != run_name:
        return False
    if dataset is not None and job.dataset != dataset:
        return False
    if model_subfolder is not None and job.model_subfolder != model_subfolder:
        return False
    if run_index is not None:
        return False
    return True


def _index_runs(
    runs: Sequence[DiscoveredRun],
) -> tuple[
    dict[str, DiscoveredRun],
    dict[tuple[str, str, str], list[DiscoveredRun]],
]:
    by_id: dict[str, DiscoveredRun] = {}
    by_semantic: dict[tuple[str, str, str], list[DiscoveredRun]] = {}
    for run in runs:
        by_id[run.metadata.run_id] = run
        key = (run.metadata.run_name, run.dataset, run.model_subfolder)
        by_semantic.setdefault(key, []).append(run)
    return by_id, by_semantic


def _build_job_only_row(
    record: DiscoveredJob,
    *,
    metrics_cache: dict[Path, tuple[float | None, float | None]],
) -> _UnifiedRow:
    _ = metrics_cache  # reserved for future queue-only metric enrichment
    job = record.job
    return _UnifiedRow(
        job_ref=job.selector or job.job_id,
        run_id=job.run_id or "-",
        dataset=job.dataset or "-",
        model_subfolder=job.model_subfolder or "-",
        name=job.run_name or Path(job.config).name,
        run_idx="-",
        state=job.status,
        status_key=job.status,
        epoch="-",
        eta="-",
        loss="-",
        failure=_short_failure_for_job(job),
        sort_time=job.updated_at,
    )


def _build_merged_row(
    record: DiscoveredJob,
    run: DiscoveredRun,
    *,
    metrics_cache: dict[Path, tuple[float | None, float | None]],
    failure_cache: dict[Path, str | None],
) -> _UnifiedRow:
    job = record.job
    train_loss, val_loss = _latest_losses_for_run(run, metrics_cache=metrics_cache)
    run_failure = _short_failure_for_run(run, failure_cache=failure_cache)
    if run_failure == "-":
        run_failure = _short_failure_for_job(job)
    return _UnifiedRow(
        job_ref=job.selector or job.job_id,
        run_id=run.metadata.run_id,
        dataset=run.dataset,
        model_subfolder=run.model_subfolder,
        name=run.metadata.run_name,
        run_idx=str(run.metadata.run_index),
        state=display_run_status(run),
        status_key=run.metadata.status,
        epoch=_format_run_epoch(run),
        eta=_format_run_eta_left(run),
        loss=_format_losses(train_loss, val_loss),
        failure=run_failure,
        sort_time=max(job.updated_at, run.last_seen),
    )


def _build_run_only_row(
    run: DiscoveredRun,
    *,
    metrics_cache: dict[Path, tuple[float | None, float | None]],
    failure_cache: dict[Path, str | None],
) -> _UnifiedRow:
    train_loss, val_loss = _latest_losses_for_run(run, metrics_cache=metrics_cache)
    return _UnifiedRow(
        job_ref="-",
        run_id=run.metadata.run_id,
        dataset=run.dataset,
        model_subfolder=run.model_subfolder,
        name=run.metadata.run_name,
        run_idx=str(run.metadata.run_index),
        state=display_run_status(run),
        status_key=run.metadata.status,
        epoch=_format_run_epoch(run),
        eta=_format_run_eta_left(run),
        loss=_format_losses(train_loss, val_loss),
        failure=_short_failure_for_run(run, failure_cache=failure_cache),
        sort_time=run.last_seen,
    )


def _row_sort_key(row: _UnifiedRow) -> tuple[int, float]:
    return (_status_bucket(row), -row.sort_time.timestamp())


def _status_bucket(row: _UnifiedRow) -> int:
    status = row.status_key
    if status in {"completed", "stopped", "failed", "done", "blocked"}:
        return 0
    if status in {"running", "pause_requested", "paused", "resuming"}:
        return 1
    if status == "queued":
        return 2
    lowered = row.state.strip().lower()
    if lowered.startswith("stale") or lowered == "crash":
        return 1
    return 0


def _render_unified_rows(rows: Sequence[_UnifiedRow], *, full: bool) -> str:
    headers = [
        "job",
        "dataset",
        "subfolder",
        "name",
        "run_idx",
        "status",
        "epoch",
        "eta",
    ]
    if full:
        headers.extend(["run_id", "loss", "failure"])

    values: list[list[str]] = []
    for row in rows:
        line = [
            row.job_ref,
            row.dataset,
            row.model_subfolder,
            row.name,
            row.run_idx,
            row.state,
            row.epoch,
            row.eta,
        ]
        if full:
            line.extend([row.run_id, row.loss, row.failure])
        values.append(line)

    widths = [max(len(headers[idx]), *(len(item[idx]) for item in values)) for idx in range(len(headers))]
    lines = [
        "  ".join(headers[idx].ljust(widths[idx]) for idx in range(len(headers))),
        "  ".join("-" * widths[idx] for idx in range(len(headers))),
    ]
    lines.extend("  ".join(item[idx].ljust(widths[idx]) for idx in range(len(headers))) for item in values)
    return "\n".join(lines)


def _show_job_details(record: DiscoveredJob, *, stdout) -> int:
    job = record.job
    scan_result = scan_runs()
    run_by_id, run_by_semantic = _index_runs(scan_result.runs)
    linked = _resolve_run_for_job(job, run_by_id=run_by_id, run_by_semantic=run_by_semantic)
    metrics_cache: dict[Path, tuple[float | None, float | None]] = {}
    train_loss, val_loss = (None, None)
    epoch = "-"
    eta = "-"
    if linked is not None:
        train_loss, val_loss = _latest_losses_for_run(linked, metrics_cache=metrics_cache)
        epoch = _format_run_epoch(linked)
        eta = _format_run_eta_left(linked)

    print("kind          : queue_job", file=stdout)
    print(f"selector      : {job.selector or '-'}", file=stdout)
    print(f"job_id        : {job.job_id}", file=stdout)
    print(f"status        : {job.status}", file=stdout)
    print(f"priority      : {job.priority}", file=stdout)
    print(f"dataset       : {job.dataset or '-'}", file=stdout)
    print(f"subfolder     : {job.model_subfolder or '-'}", file=stdout)
    print(f"run_name      : {job.run_name or '-'}", file=stdout)
    print(f"run_id        : {job.run_id or '-'}", file=stdout)
    print(f"progress      : {epoch}", file=stdout)
    print(f"eta_left      : {eta}", file=stdout)
    print(f"latest_loss   : {_format_losses(train_loss, val_loss)}", file=stdout)
    print(f"failure       : {_short_failure_for_job(job)}", file=stdout)
    print(f"submitted_at  : {format_timestamp_local(job.submitted_at)}", file=stdout)
    print(
        f"launched_at   : {'-' if job.launched_at is None else format_timestamp_local(job.launched_at)}",
        file=stdout,
    )
    print(
        f"finished_at   : {'-' if job.finished_at is None else format_timestamp_local(job.finished_at)}",
        file=stdout,
    )
    print(f"job_file      : {record.path}", file=stdout)
    print(f"config        : {job.config}", file=stdout)
    print(f"source_config : {job.source_config or '-'}", file=stdout)
    print(f"log_path      : {_resolve_canonical_job_log_path(job)}", file=stdout)
    if linked is not None:
        print(f"run_dir       : {linked.run_dir}", file=stdout)
        print(f"run_meta      : {linked.metadata_path}", file=stdout)
    return 0


def _show_run_details(run: DiscoveredRun, *, stdout) -> int:
    metrics_cache: dict[Path, tuple[float | None, float | None]] = {}
    failure_cache: dict[Path, str | None] = {}
    train_loss, val_loss = _latest_losses_for_run(run, metrics_cache=metrics_cache)
    print("kind          : run", file=stdout)
    print(f"run_id        : {run.metadata.run_id}", file=stdout)
    print(f"run_ref       : {run.dataset}/{run.model_subfolder}/{run.run_dir.name}", file=stdout)
    print(f"status        : {display_run_status(run)}", file=stdout)
    print(f"dataset       : {run.dataset}", file=stdout)
    print(f"subfolder     : {run.model_subfolder}", file=stdout)
    print(f"run_name      : {run.metadata.run_name}", file=stdout)
    print(f"run_index     : {run.metadata.run_index}", file=stdout)
    print(f"progress      : {_format_run_epoch(run)}", file=stdout)
    print(f"eta_left      : {_format_run_eta_left(run)}", file=stdout)
    print(f"latest_loss   : {_format_losses(train_loss, val_loss)}", file=stdout)
    print(f"best_val_loss : {'-' if run.metadata.best_val_loss is None else f'{run.metadata.best_val_loss:.6g}'}", file=stdout)
    print(f"failure       : {_short_failure_for_run(run, failure_cache=failure_cache)}", file=stdout)
    print(f"created_at    : {format_timestamp_local(run.metadata.created_at)}", file=stdout)
    print(f"last_seen     : {format_timestamp_local(run.last_seen)}", file=stdout)
    print(
        f"ended_at      : {'-' if run.metadata.ended_at is None else format_timestamp_local(run.metadata.ended_at)}",
        file=stdout,
    )
    print(f"run_dir       : {run.run_dir}", file=stdout)
    print(f"metadata_path : {run.metadata_path}", file=stdout)
    print(f"log_path      : {_resolve_canonical_run_log_path(run)}", file=stdout)
    print(f"loss_path     : {_PATHS.loss_file_path(run_dir=run.run_dir).resolve()}", file=stdout)
    return 0


def _resolve_target(
    token: str,
    *,
    stdout,
    stderr,
    stdin,
) -> _ResolvedTarget | None:
    normalized = token.strip()
    if not normalized:
        print("Empty selector is not supported.", file=stderr)
        return None

    queue_records, invalid_jobs = discover_jobs()
    for item in invalid_jobs:
        print(
            f"warning: skipped invalid queue job {item.path} ({item.kind}: {item.message})",
            file=stderr,
        )

    scan_result = scan_runs()
    write_invalid_run_warnings(scan_result.invalid, stderr=stderr)

    queue_matches = _match_queue_target(list(queue_records), normalized)
    run_matches = _match_run_target(scan_result.runs, normalized)

    candidates = [*(_ResolvedTarget(kind="job", job=item) for item in queue_matches)]
    candidates.extend(_ResolvedTarget(kind="run", run=item) for item in run_matches)
    if not candidates:
        print(f"No matching job or run found for {normalized!r}.", file=stderr)
        print(
            "Use qNNNN/job_id for queue jobs, or run_id/dataset[/subfolder]/run_dir_name for runs.",
            file=stderr,
        )
        return None

    selected = resolve_ambiguous_matches(
        candidates,
        render_matches=_render_candidate_table,
        heading="Multiple jobs/runs match this selector:",
        rerun_hint=(
            "Rerun with an exact qNNNN/job_id/run_id selector, or with dataset[/subfolder]/run_dir_name."
        ),
        selection_name="entry",
        stdin=stdin,
        stdout=stdout,
        stderr=stderr,
    )
    if selected is None:
        return None
    return selected


def _render_candidate_table(candidates: Sequence[_ResolvedTarget]) -> str:
    headers = ["#", "kind", "id", "dataset", "subfolder", "name", "state"]
    width = max(2, len(str(len(candidates))))
    rows = [
        [
            f"{index:0{width}d}",
            candidate.kind,
            candidate.id_text,
            candidate.dataset,
            candidate.model_subfolder,
            candidate.name,
            candidate.state,
        ]
        for index, candidate in enumerate(candidates, start=1)
    ]
    widths = [max(len(headers[idx]), *(len(row[idx]) for row in rows)) for idx in range(len(headers))]
    lines = [
        "  ".join(headers[idx].ljust(widths[idx]) for idx in range(len(headers))),
        "  ".join("-" * widths[idx] for idx in range(len(headers))),
    ]
    lines.extend("  ".join(row[idx].ljust(widths[idx]) for idx in range(len(headers))) for row in rows)
    return "\n".join(lines)


def _match_queue_target(records: list[DiscoveredJob], token: str) -> list[DiscoveredJob]:
    lowered = token.lower()

    exact_selector = [record for record in records if (record.job.selector or "").lower() == lowered]
    if exact_selector:
        return exact_selector

    exact_job_id = [record for record in records if record.job.job_id.lower() == lowered]
    if exact_job_id:
        return exact_job_id

    if is_queue_selector(token):
        return []

    prefix_matches = [record for record in records if record.job.job_id.lower().startswith(lowered)]
    if prefix_matches:
        return prefix_matches
    return []


def _match_run_target(runs: Sequence[DiscoveredRun], token: str) -> list[DiscoveredRun]:
    lowered = token.lower()
    if "/" in token:
        try:
            dataset, model_subfolder, run_dir_name = _parse_run_ref_selector(token)
        except ValueError:
            return []
        return [
            run
            for run in runs
            if run.dataset == dataset
            and run.model_subfolder == model_subfolder
            and run.run_dir.name == run_dir_name
        ]

    exact_run_id = [run for run in runs if run.metadata.run_id.lower() == lowered]
    if exact_run_id:
        return exact_run_id

    run_id_prefix = [run for run in runs if run.metadata.run_id.lower().startswith(lowered)]
    if run_id_prefix:
        return run_id_prefix

    exact_run_name = [run for run in runs if run.metadata.run_name == token]
    if exact_run_name:
        return exact_run_name

    run_dir_matches = [run for run in runs if run.run_dir.name == token]
    if run_dir_matches:
        return run_dir_matches
    return []


def _parse_run_ref_selector(run_ref: str) -> tuple[str, str, str]:
    parts = [part for part in run_ref.replace("\\", "/").split("/") if part]
    if len(parts) < 2:
        raise ValueError(
            "Run reference must be 'dataset/run_dir_name' or 'dataset/subfolder/run_dir_name'."
        )
    dataset_name = parts[0]
    run_dir_name = parts[-1]
    model_subfolder = "/".join(parts[1:-1])
    return dataset_name, model_subfolder, run_dir_name


def _resolve_run_for_job(
    job,
    *,
    run_by_id: dict[str, DiscoveredRun],
    run_by_semantic: dict[tuple[str, str, str], list[DiscoveredRun]],
) -> DiscoveredRun | None:
    if job.run_id is not None:
        match = run_by_id.get(job.run_id)
        if match is not None:
            return match

    if job.run_name and job.dataset and job.model_subfolder:
        candidates = run_by_semantic.get((job.run_name, job.dataset, job.model_subfolder), [])
        if len(candidates) == 1:
            return candidates[0]
    return None


def _latest_losses_for_run(
    run: DiscoveredRun,
    *,
    metrics_cache: dict[Path, tuple[float | None, float | None]],
) -> tuple[float | None, float | None]:
    loss_path = _PATHS.loss_file_path(run_dir=run.run_dir).resolve()
    cached = metrics_cache.get(loss_path)
    if cached is not None:
        return cached

    train_loss: float | None = None
    val_loss: float | None = None
    if loss_path.exists():
        try:
            with loss_path.open("r", encoding="utf-8", errors="replace") as handle:
                lines = [line.strip() for line in handle if line.strip()]
            for line in reversed(lines):
                tokens = [item for item in line.split() if item]
                if len(tokens) < 3:
                    continue
                try:
                    train_loss = float(tokens[1])
                    val_loss = float(tokens[2])
                    break
                except ValueError:
                    continue
        except OSError:
            train_loss = None
            val_loss = None

    metrics_cache[loss_path] = (train_loss, val_loss)
    return train_loss, val_loss


def _short_failure_for_job(job) -> str:
    if job.status not in {"failed", "blocked"}:
        return "-"
    return _single_line(job.error or "-")


def _short_failure_for_run(
    run: DiscoveredRun,
    *,
    failure_cache: dict[Path, str | None],
) -> str:
    if run.metadata.status != "failed":
        return "-"

    if run.metadata.failure_reason:
        return _single_line(run.metadata.failure_reason)

    log_path = _resolve_canonical_run_log_path(run)
    cached = failure_cache.get(log_path)
    if cached is not None:
        return _single_line(cached) if cached else "-"

    hint = _latest_error_hint(log_path)
    failure_cache[log_path] = hint
    return _single_line(hint) if hint else "-"


def _single_line(value: str) -> str:
    text = value.replace("\n", " ").replace("\r", " ").strip()
    if len(text) <= 120:
        return text
    return f"{text[:117]}..."


def _format_run_epoch(run: DiscoveredRun) -> str:
    if run.metadata.last_epoch is None and run.metadata.max_epoch is None:
        return "-"
    last_epoch = "-" if run.metadata.last_epoch is None else str(run.metadata.last_epoch + 1)
    max_epoch = "-" if run.metadata.max_epoch is None else str(run.metadata.max_epoch)
    return f"{last_epoch}/{max_epoch}"


def _format_run_eta_left(run: DiscoveredRun) -> str:
    seconds_left = _estimate_seconds_left(run)
    if seconds_left is None:
        return "-"

    total_minutes = max(1, int((seconds_left + 59.0) // 60.0))
    days, rem_minutes = divmod(total_minutes, 24 * 60)
    hours, minutes = divmod(rem_minutes, 60)
    return f"{days}d{hours}h{minutes}m"


def _estimate_seconds_left(run: DiscoveredRun) -> float | None:
    metadata = run.metadata
    if metadata.status != "running" or metadata.closed_cleanly:
        return None
    if metadata.max_epoch is None:
        return None

    completed_epochs = 0 if metadata.last_epoch is None else metadata.last_epoch + 1
    remaining_epochs = metadata.max_epoch - completed_epochs
    if remaining_epochs <= 0:
        return None

    mean_epoch_duration_s = _mean_epoch_duration_seconds(run)
    if mean_epoch_duration_s is None or mean_epoch_duration_s <= 0:
        return None
    return float(remaining_epochs) * mean_epoch_duration_s


def _mean_epoch_duration_seconds(run: DiscoveredRun) -> float | None:
    live_stats = run.metadata.live_runtime_stats
    if live_stats is not None:
        recent = [float(value) for value in live_stats.recent_epoch_durations_s]
        if recent:
            return float(sum(recent) / len(recent))
        if live_stats.last_epoch_duration_s is not None:
            return float(live_stats.last_epoch_duration_s)
        if live_stats.median_epoch_duration_s is not None:
            return float(live_stats.median_epoch_duration_s)

    runtime_stats = run.metadata.runtime_stats
    if runtime_stats is not None and runtime_stats.training_time_per_epoch_sec is not None:
        return float(runtime_stats.training_time_per_epoch_sec)
    return None


def _format_losses(train_loss: float | None, val_loss: float | None) -> str:
    parts: list[str] = []
    if train_loss is not None:
        parts.append(f"t={train_loss:.6g}")
    if val_loss is not None:
        parts.append(f"v={val_loss:.6g}")
    if not parts:
        return "-"
    return " ".join(parts)


def _latest_error_hint(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            lines = [line.rstrip("\n") for line in deque(handle, maxlen=400)]
    except OSError:
        return None
    for line in reversed(lines):
        if _looks_like_error_line(line):
            return line.strip()
    return None


def _resolve_canonical_job_log_path(job) -> Path:
    # Canonical queue log rule:
    # 1) If queue metadata persisted `job.log_path`, use it.
    # 2) Otherwise fallback to queue/logs/job_<selector>_<job_id>.log naming.
    if job.log_path:
        return Path(job.log_path).expanduser().resolve()
    return (queue_logs_dir() / job_log_filename(job.job_id, selector=job.selector)).resolve()


def _resolve_canonical_run_log_path(run: DiscoveredRun) -> Path:
    # Canonical run log rule:
    # Use the run artifact path configured via Paths.log_file_path(run_dir=...),
    # which maps to project run_layout.artifacts.train_log.
    return _PATHS.log_file_path(run_dir=run.run_dir).resolve()


def _tail_lines(path: Path, *, limit: int) -> list[str]:
    if limit <= 0:
        return []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return [line.rstrip("\n") for line in deque(handle, maxlen=limit)]


def _follow_log(path: Path, *, stdout) -> None:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        handle.seek(0, os.SEEK_END)
        while True:
            line = handle.readline()
            if not line:
                time.sleep(0.5)
                continue
            print(line.rstrip("\n"), file=stdout, flush=True)


def _looks_like_error_line(line: str) -> bool:
    lowered = line.lower()
    return any(token in lowered for token in ERROR_LINE_HINTS)


__all__ = [
    "UNIFIED_STATUS_CHOICES",
    "add_list_subparser",
    "add_logs_subparser",
    "add_show_subparser",
    "list_targets",
    "logs_target",
    "show_target",
]
