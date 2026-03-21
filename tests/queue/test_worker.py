from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import lisai.queue.worker as worker_mod
import lisai.runs.listing as runs_listing
from lisai.config import settings
from lisai.runs.scanner import DiscoveredRun, ScanResults
from lisai.runs.schema import RunMetadata, TrainingSignature

from lisai.queue.state import create_queued_job
from lisai.queue.storage import discover_jobs
from lisai.queue.worker import QueueWorker


def _run(
    *,
    run_id: str,
    status: str,
    closed_cleanly: bool,
    last_heartbeat_at: str,
    signature: TrainingSignature | None,
    peak_gpu_mem_mb: int | None,
) -> DiscoveredRun:
    metadata = RunMetadata.model_validate(
        {
            "schema_version": 2,
            "run_id": run_id,
            "run_name": "demo",
            "run_index": 0,
            "dataset": "Gag",
            "model_subfolder": "HDN",
            "status": status,
            "closed_cleanly": closed_cleanly,
            "created_at": "2026-03-20T10:00:00Z",
            "updated_at": "2026-03-20T11:00:00Z",
            "ended_at": None if status == "running" else "2026-03-20T11:00:00Z",
            "last_heartbeat_at": last_heartbeat_at,
            "last_epoch": 10,
            "max_epoch": 10,
            "best_val_loss": 0.1,
            "path": "datasets/Gag/models/HDN/demo_00",
            "group_path": None,
            "training_signature": None if signature is None else signature.model_dump(mode="json"),
            "runtime_stats": None if peak_gpu_mem_mb is None else {"peak_gpu_mem_mb": peak_gpu_mem_mb},
        }
    )
    run_dir = Path("/tmp/Gag/demo_00")
    return DiscoveredRun(
        metadata=metadata,
        metadata_path=run_dir / ".lisai_run_meta.json",
        run_dir=run_dir,
        dataset="Gag",
        model_subfolder="HDN",
        group_path=None,
        path="datasets/Gag/models/HDN/demo_00",
        path_consistent=True,
        consistency_issues=(),
    )


def _queued_job(tmp_path: Path, *, resource_class: str = "medium"):
    return create_queued_job(
        config_path=tmp_path / "cfg.yml",
        resource_class=resource_class,
        device="cuda:0",
        queue_root=tmp_path / ".lisai" / "queue",
        dataset="Gag",
        model_subfolder="HDN",
        run_name="demo",
        training_signature=TrainingSignature(architecture="unet", batch_size=8, patch_size=128),
        now=datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc),
    )


def test_worker_launches_job_when_history_and_free_vram_allow(monkeypatch, tmp_path):
    _queued_job(tmp_path)
    matching_signature = TrainingSignature(architecture="unet", batch_size=8, patch_size=128)
    past_run = _run(
        run_id="01ARZ3NDEKTSV4RRFFQ69G5FAA",
        status="completed",
        closed_cleanly=True,
        last_heartbeat_at="2026-03-20T11:00:00Z",
        signature=matching_signature,
        peak_gpu_mem_mb=5000,
    )

    monkeypatch.setattr(worker_mod, "scan_runs", lambda: ScanResults(runs=(past_run,), invalid=()))
    monkeypatch.setattr(worker_mod, "query_free_vram_mb", lambda device: 7000)

    launched: list[str] = []

    def fake_launch(self, record):
        launched.append(record.job.job_id)
        return True

    monkeypatch.setattr(QueueWorker, "_launch_job", fake_launch)
    worker = QueueWorker(queue_root=tmp_path / ".lisai" / "queue", safety_margin_mb=1000)
    worker.run_once()

    assert launched


def test_worker_keeps_job_queued_when_free_vram_is_insufficient(monkeypatch, tmp_path):
    created = _queued_job(tmp_path, resource_class="heavy")
    matching_signature = TrainingSignature(architecture="unet", batch_size=8, patch_size=128)
    past_run = _run(
        run_id="01ARZ3NDEKTSV4RRFFQ69G5FAB",
        status="completed",
        closed_cleanly=True,
        last_heartbeat_at="2026-03-20T11:00:00Z",
        signature=matching_signature,
        peak_gpu_mem_mb=5000,
    )

    monkeypatch.setattr(worker_mod, "scan_runs", lambda: ScanResults(runs=(past_run,), invalid=()))
    monkeypatch.setattr(worker_mod, "query_free_vram_mb", lambda device: 5500)

    launched: list[str] = []
    monkeypatch.setattr(QueueWorker, "_launch_job", lambda self, record: launched.append(record.job.job_id) or True)

    worker = QueueWorker(queue_root=tmp_path / ".lisai" / "queue", safety_margin_mb=1000)
    worker.run_once()

    assert launched == []
    queued_records, _ = discover_jobs(status="queued", queue_root=tmp_path / ".lisai" / "queue")
    assert len(queued_records) == 1
    assert queued_records[0].job.job_id == created.job.job_id


def test_worker_treats_old_running_heartbeat_as_stale_not_active(monkeypatch, tmp_path):
    _queued_job(tmp_path)
    stale_run = _run(
        run_id="01ARZ3NDEKTSV4RRFFQ69G5FAC",
        status="running",
        closed_cleanly=False,
        last_heartbeat_at="2026-03-20T10:00:00Z",
        signature=None,
        peak_gpu_mem_mb=None,
    )

    monkeypatch.setattr(worker_mod, "scan_runs", lambda: ScanResults(runs=(stale_run,), invalid=()))
    monkeypatch.setattr(worker_mod, "query_free_vram_mb", lambda device: 10000)
    monkeypatch.setattr(settings.project.run_tracking, "active_heartbeat_timeout_minutes", 10)
    monkeypatch.setattr(runs_listing, "utc_now", lambda: datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc))

    captured_active_counts: list[int] = []
    original = QueueWorker._launch_decision

    def wrapped(self, job, runs, *, active_runs):
        captured_active_counts.append(len(active_runs))
        return original(self, job, runs, active_runs=active_runs)

    monkeypatch.setattr(QueueWorker, "_launch_decision", wrapped)
    monkeypatch.setattr(QueueWorker, "_launch_job", lambda self, record: True)

    worker = QueueWorker(queue_root=tmp_path / ".lisai" / "queue", safety_margin_mb=1000)
    worker.run_once()

    assert captured_active_counts == [0]
