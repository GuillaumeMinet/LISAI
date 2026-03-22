from __future__ import annotations

import io
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import lisai.queue.cli as queue_cli
from lisai.cli import main as root_main
from lisai.queue.history import SchedulingContext
from lisai.queue.schema import QueueJob
from lisai.queue.storage import DiscoveredJob, discover_jobs, queue_logs_dir, queue_state_dir, write_job_atomic
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
    assert "id" in listed.out
    assert "name" in listed.out
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
    exit_code = queue_cli.clean_jobs(older_than="7d", status=None, clean_all=False)
    assert exit_code == 0

    done_jobs, _ = discover_jobs(status="done", queue_root=queue_root)
    failed_jobs, _ = discover_jobs(status="failed", queue_root=queue_root)
    assert done_jobs == ()
    assert len(failed_jobs) == 1
    assert failed_jobs[0].job.job_id == "job_recent_failed"


def test_queue_list_status_filter(monkeypatch, tmp_path, capsys):
    queue_root = tmp_path / ".lisai" / "queue"
    monkeypatch.setenv("LISAI_QUEUE_ROOT", str(queue_root))
    now = datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc)

    queued_job = QueueJob(
        job_id="job_queued",
        selector="q0001",
        config=str(tmp_path / "queued.yml"),
        status="queued",
        device="cuda:0",
        submitted_at=now - timedelta(minutes=10),
        updated_at=now - timedelta(minutes=10),
        resource_class="medium",
    )
    done_job = QueueJob(
        job_id="job_done",
        selector="q0002",
        config=str(tmp_path / "done.yml"),
        status="done",
        device="cuda:0",
        submitted_at=now - timedelta(hours=2),
        updated_at=now - timedelta(hours=1),
        resource_class="medium",
        finished_at=now - timedelta(hours=1),
        exit_code=0,
    )
    write_job_atomic(queue_state_dir("queued", queue_root=queue_root) / "job_q0001_job_queued.json", queued_job)
    write_job_atomic(queue_state_dir("done", queue_root=queue_root) / "job_q0002_job_done.json", done_job)

    exit_code = root_main(["queue", "list", "--status", "queued"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "q0001" in captured.out
    assert "job_done" not in captured.out


def test_queue_list_uses_local_timestamp_formatter(monkeypatch, tmp_path, capsys):
    queue_root = tmp_path / ".lisai" / "queue"
    monkeypatch.setenv("LISAI_QUEUE_ROOT", str(queue_root))
    now = datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc)

    queued_job = QueueJob(
        job_id="job_with_local_ts",
        selector="q0004",
        config=str(tmp_path / "queued.yml"),
        status="queued",
        device="cuda:0",
        submitted_at=now - timedelta(minutes=10),
        updated_at=now - timedelta(minutes=10),
        resource_class="medium",
    )
    write_job_atomic(
        queue_state_dir("queued", queue_root=queue_root) / "job_q0004_job_with_local_ts.json",
        queued_job,
    )

    monkeypatch.setattr(queue_cli, "format_timestamp_local", lambda _value: "LOCAL_TS")

    exit_code = root_main(["queue", "list", "--status", "queued"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "LOCAL_TS" in captured.out


def test_queue_clean_removes_associated_logs(monkeypatch, tmp_path):
    queue_root = tmp_path / ".lisai" / "queue"
    monkeypatch.setenv("LISAI_QUEUE_ROOT", str(queue_root))
    old_time = datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc)

    done_job = QueueJob(
        job_id="job_done_for_log_cleanup",
        selector="q0007",
        config=str(tmp_path / "done.yml"),
        status="done",
        device="cuda:0",
        submitted_at=old_time - timedelta(hours=1),
        updated_at=old_time,
        resource_class="medium",
        finished_at=old_time,
        exit_code=0,
    )
    write_job_atomic(
        queue_state_dir("done", queue_root=queue_root) / "job_q0007_job_done_for_log_cleanup.json",
        done_job,
    )
    log_path = queue_logs_dir(queue_root=queue_root) / "job_q0007_job_done_for_log_cleanup.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("hello\n", encoding="utf-8")

    monkeypatch.setattr(queue_cli, "utc_now", lambda: datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc))
    exit_code = queue_cli.clean_jobs(older_than="7d", status=None, clean_all=False)
    assert exit_code == 0
    assert not log_path.exists()


def test_queue_cancel_queued_job_by_selector(monkeypatch, tmp_path):
    queue_root = tmp_path / ".lisai" / "queue"
    monkeypatch.setenv("LISAI_QUEUE_ROOT", str(queue_root))
    now = datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc)

    queued_job = QueueJob(
        job_id="job_cancel_me",
        selector="q0003",
        config=str(tmp_path / "queued.yml"),
        status="queued",
        device="cuda:0",
        submitted_at=now - timedelta(minutes=5),
        updated_at=now - timedelta(minutes=5),
        resource_class="medium",
    )
    write_job_atomic(
        queue_state_dir("queued", queue_root=queue_root) / "job_q0003_job_cancel_me.json",
        queued_job,
    )

    exit_code = root_main(["queue", "cancel", "q0003", "--yes"])
    assert exit_code == 0
    queued_jobs, _ = discover_jobs(status="queued", queue_root=queue_root)
    failed_jobs, _ = discover_jobs(status="failed", queue_root=queue_root)
    assert queued_jobs == ()
    assert len(failed_jobs) == 1
    assert failed_jobs[0].job.job_id == "job_cancel_me"
    assert failed_jobs[0].job.error == "cancelled_by_user(queued)"


def test_cancel_eval_policy_follows_config_default(monkeypatch, tmp_path):
    job = QueueJob(
        job_id="job_eval_policy",
        selector="q0009",
        config=str(tmp_path / "cfg.yml"),
        status="running",
        device="cuda:0",
        submitted_at=datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc),
        updated_at=datetime(2026, 3, 20, 12, 1, tzinfo=timezone.utc),
        resource_class="medium",
        launched_at=datetime(2026, 3, 20, 12, 1, tzinfo=timezone.utc),
    )

    monkeypatch.setattr(
        queue_cli,
        "resolve_config",
        lambda _path: SimpleNamespace(experiment=SimpleNamespace(post_training_inference=True)),
    )
    assert queue_cli._should_evaluate_after_cancel(job, eval_override=None, stderr=io.StringIO()) is True

    monkeypatch.setattr(
        queue_cli,
        "resolve_config",
        lambda _path: SimpleNamespace(experiment=SimpleNamespace(post_training_inference=False)),
    )
    assert queue_cli._should_evaluate_after_cancel(job, eval_override=None, stderr=io.StringIO()) is False


def test_cancel_eval_policy_override_takes_priority(monkeypatch, tmp_path):
    job = QueueJob(
        job_id="job_eval_override",
        selector="q0010",
        config=str(tmp_path / "cfg.yml"),
        status="running",
        device="cuda:0",
        submitted_at=datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc),
        updated_at=datetime(2026, 3, 20, 12, 1, tzinfo=timezone.utc),
        resource_class="medium",
        launched_at=datetime(2026, 3, 20, 12, 1, tzinfo=timezone.utc),
    )

    monkeypatch.setattr(
        queue_cli,
        "resolve_config",
        lambda _path: SimpleNamespace(experiment=SimpleNamespace(post_training_inference=False)),
    )
    assert queue_cli._should_evaluate_after_cancel(job, eval_override=True, stderr=io.StringIO()) is True
    assert queue_cli._should_evaluate_after_cancel(job, eval_override=False, stderr=io.StringIO()) is False


def test_cancel_running_process_not_found_triggers_eval_hook(monkeypatch, tmp_path):
    queue_root = tmp_path / ".lisai" / "queue"
    monkeypatch.setenv("LISAI_QUEUE_ROOT", str(queue_root))
    now = datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc)

    running_job = QueueJob(
        job_id="job_cancel_running_eval",
        selector="q0011",
        config=str(tmp_path / "cfg.yml"),
        status="running",
        device="cuda:0",
        submitted_at=now - timedelta(minutes=10),
        updated_at=now - timedelta(minutes=1),
        resource_class="medium",
        launched_at=now - timedelta(minutes=9),
        pid=424242,
    )
    write_job_atomic(
        queue_state_dir("running", queue_root=queue_root) / "job_q0011_job_cancel_running_eval.json",
        running_job,
    )
    records, _invalid = discover_jobs(status="running", queue_root=queue_root)
    assert len(records) == 1
    record: DiscoveredJob = records[0]

    called: list[tuple[str, bool | None]] = []
    monkeypatch.setattr(queue_cli, "_pid_exists", lambda _pid: False)
    monkeypatch.setattr(
        queue_cli,
        "_maybe_run_post_cancel_evaluation",
        lambda job, *, eval_override, stdout, stderr: called.append((job.job_id, eval_override)),
    )

    ok = queue_cli._cancel_single_record(
        record,
        force=False,
        allow_force_prompt=False,
        eval_override=True,
        stdin=io.StringIO(""),
        stdout=io.StringIO(),
        stderr=io.StringIO(),
    )
    assert ok is True
    assert called == [("job_cancel_running_eval", True)]
