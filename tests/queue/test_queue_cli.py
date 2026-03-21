from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import lisai.queue.cli as queue_cli
from lisai.cli import main as root_main
from lisai.queue.history import SchedulingContext
from lisai.queue.schema import QueueJob
from lisai.queue.storage import discover_jobs, queue_state_dir, write_job_atomic
from lisai.runs.schema import TrainingSignature


def test_queue_submit_and_list_via_root_cli(monkeypatch, tmp_path, capsys):
    queue_root = tmp_path / ".lisai" / "queue"
    config_path = tmp_path / "cfg.yml"
    config_path.write_text("{}", encoding="utf-8")

    monkeypatch.setenv("LISAI_QUEUE_ROOT", str(queue_root))
    monkeypatch.setattr(queue_cli, "resolve_config_path", lambda _: config_path)
    monkeypatch.setattr(
        queue_cli,
        "load_scheduling_context",
        lambda _: SchedulingContext(
            config_path=config_path,
            dataset="Gag",
            model_subfolder="HDN",
            run_name="demo",
            training_signature=TrainingSignature(architecture="unet", batch_size=8, patch_size=128),
        ),
    )

    submit_exit = root_main(["queue", "submit", "--config", "cfg.yml", "--resource-class", "light"])
    submit_captured = capsys.readouterr()
    assert submit_exit == 0
    assert "Submitted job" in submit_captured.out

    list_exit = root_main(["queue", "list"])
    listed = capsys.readouterr()
    assert list_exit == 0
    assert "job_id" in listed.out
    assert "status" in listed.out
    assert "queued" in listed.out


def test_queue_clean_removes_old_done_and_failed_jobs(monkeypatch, tmp_path):
    queue_root = tmp_path / ".lisai" / "queue"
    monkeypatch.setenv("LISAI_QUEUE_ROOT", str(queue_root))

    old_time = datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc)
    recent_time = datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc)

    old_done = QueueJob(
        job_id="job_old_done",
        config=str(tmp_path / "a.yml"),
        status="done",
        device="cuda:0",
        submitted_at=old_time - timedelta(hours=1),
        updated_at=old_time,
        resource_class="medium",
        finished_at=old_time,
        exit_code=0,
    )
    recent_failed = QueueJob(
        job_id="job_recent_failed",
        config=str(tmp_path / "b.yml"),
        status="failed",
        device="cuda:0",
        submitted_at=recent_time - timedelta(hours=1),
        updated_at=recent_time,
        resource_class="medium",
        finished_at=recent_time,
        exit_code=1,
        error="boom",
    )

    write_job_atomic(queue_state_dir("done", queue_root=queue_root) / "job_old_done.json", old_done)
    write_job_atomic(
        queue_state_dir("failed", queue_root=queue_root) / "job_recent_failed.json",
        recent_failed,
    )

    monkeypatch.setattr(queue_cli, "utc_now", lambda: datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc))
    exit_code = queue_cli.clean_jobs(older_than="7d")
    assert exit_code == 0

    done_jobs, _ = discover_jobs(status="done", queue_root=queue_root)
    failed_jobs, _ = discover_jobs(status="failed", queue_root=queue_root)
    assert done_jobs == ()
    assert len(failed_jobs) == 1
    assert failed_jobs[0].job.job_id == "job_recent_failed"
