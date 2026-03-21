from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import IO

from lisai.config import settings
from lisai.runs.listing import is_run_likely_active, write_invalid_run_warnings
from lisai.runs.scanner import DiscoveredRun, scan_runs
from lisai.runs.schema import utc_now

from .gpu import parse_cuda_device_index, query_free_vram_mb
from .history import estimate_expected_vram_mb, resource_class_defaults_mb
from .schema import QueueJob
from .state import mark_job_done, mark_job_failed, mark_job_running, set_job_run_id
from .storage import (
    DiscoveredJob,
    discover_jobs,
    ensure_queue_dirs,
    find_job,
    queue_logs_dir,
)


@dataclass(frozen=True)
class LaunchDecision:
    should_launch: bool
    expected_vram_mb: int
    required_vram_mb: int
    free_vram_mb: int | None
    source: str
    reason: str


@dataclass
class RunningProcess:
    record: DiscoveredJob
    process: subprocess.Popen
    log_handle: IO[bytes]


class QueueWorker:
    def __init__(
        self,
        *,
        queue_root: str | Path | None = None,
        poll_seconds: int | None = None,
        safety_margin_mb: int | None = None,
        stdout=None,
        stderr=None,
    ):
        self.queue_root = ensure_queue_dirs(queue_root=queue_root)
        self.poll_seconds = int(
            settings.project.queue.poll_seconds if poll_seconds is None else poll_seconds
        )
        self.safety_margin_mb = int(
            settings.project.queue.safety_margin_mb if safety_margin_mb is None else safety_margin_mb
        )
        self.stdout = sys.stdout if stdout is None else stdout
        self.stderr = sys.stderr if stderr is None else stderr
        self._running_processes: dict[str, RunningProcess] = {}

    def run_forever(self) -> int:
        while True:
            self.run_once()
            time.sleep(self.poll_seconds)

    def run_once(self) -> None:
        scan_result = scan_runs()
        write_invalid_run_warnings(scan_result.invalid, stderr=self.stderr)

        self._refresh_finished_processes()
        self._reconcile_running_jobs(scan_result.runs)

        queued_records, invalid_jobs = discover_jobs(status="queued", queue_root=self.queue_root)
        for invalid in invalid_jobs:
            print(
                f"warning: skipped invalid queue job {invalid.path} ({invalid.kind}: {invalid.message})",
                file=self.stderr,
            )

        active_runs = [run for run in scan_result.runs if is_run_likely_active(run)]

        for record in queued_records:
            decision = self._launch_decision(record.job, scan_result.runs, active_runs=active_runs)
            if not decision.should_launch:
                continue
            launched = self._launch_job(record)
            if launched:
                # v1 policy: launch at most one job per loop, then re-evaluate next cycle.
                break

    def _launch_decision(
        self,
        job: QueueJob,
        runs: tuple[DiscoveredRun, ...],
        *,
        active_runs: list[DiscoveredRun],
    ) -> LaunchDecision:
        expected_vram_mb, source = estimate_expected_vram_mb(
            signature=job.training_signature,
            resource_class=job.resource_class,
            runs=runs,
            resource_defaults_mb=resource_class_defaults_mb(),
        )
        required_vram_mb = expected_vram_mb + self.safety_margin_mb

        try:
            free_vram_mb = query_free_vram_mb(job.device)
        except Exception as exc:
            return LaunchDecision(
                should_launch=False,
                expected_vram_mb=expected_vram_mb,
                required_vram_mb=required_vram_mb,
                free_vram_mb=None,
                source=source,
                reason=f"gpu_query_failed:{type(exc).__name__}",
            )

        if free_vram_mb is None:
            return LaunchDecision(
                should_launch=True,
                expected_vram_mb=expected_vram_mb,
                required_vram_mb=required_vram_mb,
                free_vram_mb=None,
                source=source,
                reason=f"non_cuda_device(active_runs={len(active_runs)})",
            )

        if free_vram_mb < required_vram_mb:
            return LaunchDecision(
                should_launch=False,
                expected_vram_mb=expected_vram_mb,
                required_vram_mb=required_vram_mb,
                free_vram_mb=free_vram_mb,
                source=source,
                reason=f"insufficient_free_vram(active_runs={len(active_runs)})",
            )

        return LaunchDecision(
            should_launch=True,
            expected_vram_mb=expected_vram_mb,
            required_vram_mb=required_vram_mb,
            free_vram_mb=free_vram_mb,
            source=source,
            reason=f"safe_to_launch(active_runs={len(active_runs)})",
        )

    def _launch_job(self, record: DiscoveredJob) -> bool:
        job = record.job
        log_path = queue_logs_dir(queue_root=self.queue_root) / f"{job.job_id}.log"
        log_handle: IO[bytes] | None = None
        try:
            log_handle = log_path.open("ab")

            env = os.environ.copy()
            try:
                device_index = parse_cuda_device_index(job.device)
            except Exception:
                device_index = None
            if device_index is not None:
                env["CUDA_VISIBLE_DEVICES"] = str(device_index)

            command = [sys.executable, "-m", "lisai", "train", "--config", job.config]
            process = subprocess.Popen(
                command,
                cwd=str(settings.PROJECT_ROOT),
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            running_record = mark_job_running(
                record,
                pid=process.pid,
                log_path=log_path,
                queue_root=self.queue_root,
            )
            self._running_processes[job.job_id] = RunningProcess(
                record=running_record,
                process=process,
                log_handle=log_handle,
            )
            print(
                f"launched {job.job_id} config={job.config} pid={process.pid} device={job.device}",
                file=self.stdout,
            )
            return True
        except Exception as exc:
            if log_handle is not None:
                log_handle.close()
            mark_job_failed(
                record,
                exit_code=None,
                error=f"{type(exc).__name__}: {exc}",
                queue_root=self.queue_root,
            )
            print(
                f"warning: failed to launch {job.job_id}: {type(exc).__name__}: {exc}",
                file=self.stderr,
            )
            return False

    def _refresh_finished_processes(self) -> None:
        finished: list[str] = []
        for job_id, running in self._running_processes.items():
            exit_code = running.process.poll()
            if exit_code is None:
                continue

            running.log_handle.close()
            latest_record = find_job(job_id, queue_root=self.queue_root)
            if latest_record is not None:
                if exit_code == 0:
                    mark_job_done(
                        latest_record,
                        exit_code=exit_code,
                        queue_root=self.queue_root,
                    )
                else:
                    mark_job_failed(
                        latest_record,
                        exit_code=exit_code,
                        queue_root=self.queue_root,
                    )
            finished.append(job_id)

        for job_id in finished:
            self._running_processes.pop(job_id, None)

    def _reconcile_running_jobs(self, runs: tuple[DiscoveredRun, ...]) -> None:
        running_records, _invalid = discover_jobs(status="running", queue_root=self.queue_root)
        if not running_records:
            return

        by_run_id = {run.metadata.run_id: run for run in runs}
        for record in running_records:
            linked = record
            if linked.job.run_id is None:
                maybe_run_id = self._infer_run_id_for_job(linked.job, runs)
                if maybe_run_id is not None:
                    linked = set_job_run_id(
                        linked,
                        run_id=maybe_run_id,
                        queue_root=self.queue_root,
                    )

            if linked.job.job_id in self._running_processes:
                continue

            run_id = linked.job.run_id
            if run_id is None:
                continue
            run = by_run_id.get(run_id)
            if run is None:
                continue

            if run.metadata.status in {"completed", "stopped"} and run.metadata.closed_cleanly:
                mark_job_done(linked, exit_code=0, queue_root=self.queue_root)
            elif run.metadata.status == "failed" and run.metadata.closed_cleanly:
                mark_job_failed(linked, exit_code=1, queue_root=self.queue_root)

    def _infer_run_id_for_job(
        self,
        job: QueueJob,
        runs: tuple[DiscoveredRun, ...],
    ) -> str | None:
        if job.run_name is None or job.dataset is None or job.model_subfolder is None:
            return None

        candidates = [
            run
            for run in runs
            if run.metadata.run_name == job.run_name
            and run.dataset == job.dataset
            and run.model_subfolder == job.model_subfolder
        ]
        if not candidates:
            return None

        if job.launched_at is not None:
            oldest_allowed = job.launched_at - timedelta(minutes=10)
            candidates = [run for run in candidates if run.metadata.created_at >= oldest_allowed]
            if not candidates:
                return None

        active_candidates = [run for run in candidates if is_run_likely_active(run)]
        if len(active_candidates) == 1:
            return active_candidates[0].metadata.run_id
        if len(active_candidates) > 1:
            return None

        if len(candidates) == 1:
            return candidates[0].metadata.run_id
        return None


__all__ = ["LaunchDecision", "QueueWorker", "RunningProcess"]
