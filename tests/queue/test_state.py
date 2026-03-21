from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from lisai.runs.schema import TrainingSignature

from lisai.queue.state import create_queued_job, mark_job_done, mark_job_running
from lisai.queue.storage import discover_jobs


def test_queue_job_creation_and_state_transitions(tmp_path: Path):
    now = datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc)
    signature = TrainingSignature(architecture="unet", batch_size=8, patch_size=128)

    queued = create_queued_job(
        config_path=tmp_path / "cfg.yml",
        resource_class="medium",
        device="cuda:0",
        queue_root=tmp_path / ".lisai" / "queue",
        dataset="Gag",
        model_subfolder="HDN",
        run_name="demo",
        training_signature=signature,
        now=now,
    )

    queued_jobs, _ = discover_jobs(status="queued", queue_root=tmp_path / ".lisai" / "queue")
    assert len(queued_jobs) == 1
    assert queued_jobs[0].job.job_id == queued.job.job_id

    running = mark_job_running(
        queued_jobs[0],
        pid=1234,
        log_path=tmp_path / "worker.log",
        queue_root=tmp_path / ".lisai" / "queue",
        now=now,
    )
    assert running.status == "running"

    done = mark_job_done(
        running,
        exit_code=0,
        queue_root=tmp_path / ".lisai" / "queue",
        now=now,
    )
    assert done.status == "done"

    done_jobs, _ = discover_jobs(status="done", queue_root=tmp_path / ".lisai" / "queue")
    assert len(done_jobs) == 1
    assert done_jobs[0].job.exit_code == 0
