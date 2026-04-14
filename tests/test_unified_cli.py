from __future__ import annotations

from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path

import lisai.unified_cli as unified_cli
from lisai.cli import main as root_main
from lisai.infra.fs.run_naming import parse_run_dir_name
from lisai.queue.schema import QueueJob
from lisai.queue.storage import queue_logs_dir, queue_state_dir, write_job_atomic
from lisai.runs.io import write_run_metadata_atomic
from lisai.runs.scanner import scan_runs
from lisai.runs.schema import RunMetadata


def _write_run(
    run_dir: Path,
    *,
    run_id: str,
    dataset: str,
    model_subfolder: str,
    status: str = "running",
    failure_reason: str | None = None,
    last_epoch: int | None = 2,
    max_epoch: int | None = 10,
) -> RunMetadata:
    run_dir.mkdir(parents=True, exist_ok=True)
    run_name, run_index = parse_run_dir_name(run_dir.name)
    payload = {
        "schema_version": 2,
        "run_id": run_id,
        "run_name": run_name,
        "run_index": run_index,
        "dataset": dataset,
        "model_subfolder": model_subfolder,
        "status": status,
        "closed_cleanly": status != "running",
        "created_at": "2026-03-20T10:00:00Z",
        "updated_at": "2026-03-20T10:01:00Z",
        "ended_at": None if status == "running" else "2026-03-20T10:02:00Z",
        "last_heartbeat_at": "2026-03-20T10:01:00Z",
        "last_epoch": last_epoch,
        "max_epoch": max_epoch,
        "best_val_loss": 0.25,
        "path": f"datasets/{dataset}/models/{model_subfolder}/{run_dir.name}",
        "group_path": None,
        "failure_reason": failure_reason,
    }
    metadata = RunMetadata.model_validate(payload)
    write_run_metadata_atomic(run_dir, metadata)
    return metadata


def _write_loss(run_dir: Path, *, train_loss: float, val_loss: float) -> None:
    (run_dir / "loss.txt").write_text(
        "\n".join(
            [
                "Epoch Train_loss Val_loss",
                f"0 {train_loss + 0.2:.6f} {val_loss + 0.2:.6f}",
                f"1 {train_loss:.6f} {val_loss:.6f}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_run_log(run_dir: Path, text: str) -> None:
    (run_dir / "train_log.log").write_text(text, encoding="utf-8")


def _write_queue_job(record: QueueJob, *, queue_root: Path) -> None:
    state_path = queue_state_dir(record.status, queue_root=queue_root)
    state_path.mkdir(parents=True, exist_ok=True)
    filename = f"job_{record.selector}_{record.job_id}.json" if record.selector else f"{record.job_id}.json"
    write_job_atomic(state_path / filename, record)


def test_unified_list_default_is_minimal_and_merges_linked_job_run(monkeypatch, tmp_path, capsys):
    queue_root = tmp_path / ".lisai" / "queue"
    datasets_root = tmp_path / "datasets"
    monkeypatch.setenv("LISAI_QUEUE_ROOT", str(queue_root))
    monkeypatch.setattr(unified_cli, "scan_runs", lambda: scan_runs(datasets_root))

    running_dir = datasets_root / "Gag" / "models" / "HDN" / "unified_running_00"
    failed_dir = datasets_root / "Gag" / "models" / "HDN" / "unified_failed_00"
    running_meta = _write_run(
        running_dir,
        run_id="01ARZ3NDEKTSV4RRFFQ69G5FAA",
        dataset="Gag",
        model_subfolder="HDN",
        status="running",
    )
    _write_loss(running_dir, train_loss=1.0, val_loss=0.9)
    _write_run_log(running_dir, "train ok\n")
    _write_run(
        failed_dir,
        run_id="01ARZ3NDEKTSV4RRFFQ69G5FAB",
        dataset="Gag",
        model_subfolder="HDN",
        status="failed",
        failure_reason="nan explosion in validation step",
    )

    now = datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc)
    job_running = QueueJob(
        job_id="job_unified_running",
        selector="q0001",
        config=str(tmp_path / "cfg_a.yml"),
        status="running",
        device="cuda:0",
        submitted_at=now - timedelta(minutes=20),
        updated_at=now - timedelta(minutes=1),
        resource_class="medium",
        run_id=running_meta.run_id,
        dataset="Gag",
        model_subfolder="HDN",
        run_name="unified_running",
        launched_at=now - timedelta(minutes=19),
    )
    job_failed = QueueJob(
        job_id="job_unified_failed",
        selector="q0002",
        config=str(tmp_path / "cfg_b.yml"),
        status="failed",
        device="cuda:0",
        submitted_at=now - timedelta(minutes=30),
        updated_at=now - timedelta(minutes=2),
        resource_class="medium",
        dataset="Gag",
        model_subfolder="HDN",
        run_name="queue_failed",
        finished_at=now - timedelta(minutes=2),
        error="cuda out of memory",
    )
    _write_queue_job(job_running, queue_root=queue_root)
    _write_queue_job(job_failed, queue_root=queue_root)

    exit_code = root_main(["list"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "job" in captured.out.splitlines()[0]
    assert "run_idx" in captured.out.splitlines()[0]
    assert "loss" not in captured.out.splitlines()[0]
    assert "failure" not in captured.out.splitlines()[0]
    assert "q0001" in captured.out
    assert "unified_running" in captured.out
    # linked run+job appears once in the merged dashboard
    assert captured.out.count("unified_running") == 1


def test_unified_list_full_includes_loss_failure_and_run_id(monkeypatch, tmp_path, capsys):
    queue_root = tmp_path / ".lisai" / "queue"
    datasets_root = tmp_path / "datasets"
    monkeypatch.setenv("LISAI_QUEUE_ROOT", str(queue_root))
    monkeypatch.setattr(unified_cli, "scan_runs", lambda: scan_runs(datasets_root))

    running_dir = datasets_root / "Gag" / "models" / "HDN" / "full_running_00"
    failed_dir = datasets_root / "Gag" / "models" / "HDN" / "full_failed_00"
    running_meta = _write_run(
        running_dir,
        run_id="01ARZ3NDEKTSV4RRFFQ69G5FB1",
        dataset="Gag",
        model_subfolder="HDN",
        status="running",
    )
    _write_loss(running_dir, train_loss=1.0, val_loss=0.9)
    _write_run_log(running_dir, "train ok\n")
    _write_run(
        failed_dir,
        run_id="01ARZ3NDEKTSV4RRFFQ69G5FB2",
        dataset="Gag",
        model_subfolder="HDN",
        status="failed",
        failure_reason="nan explosion in validation step",
    )

    now = datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc)
    job_running = QueueJob(
        job_id="job_full_running",
        selector="q0010",
        config=str(tmp_path / "cfg_a.yml"),
        status="running",
        device="cuda:0",
        submitted_at=now - timedelta(minutes=20),
        updated_at=now - timedelta(minutes=1),
        resource_class="medium",
        run_id=running_meta.run_id,
        dataset="Gag",
        model_subfolder="HDN",
        run_name="full_running",
        launched_at=now - timedelta(minutes=19),
    )
    job_failed = QueueJob(
        job_id="job_full_failed",
        selector="q0011",
        config=str(tmp_path / "cfg_b.yml"),
        status="failed",
        device="cuda:0",
        submitted_at=now - timedelta(minutes=30),
        updated_at=now - timedelta(minutes=2),
        resource_class="medium",
        dataset="Gag",
        model_subfolder="HDN",
        run_name="queue_failed",
        finished_at=now - timedelta(minutes=2),
        error="cuda out of memory",
    )
    _write_queue_job(job_running, queue_root=queue_root)
    _write_queue_job(job_failed, queue_root=queue_root)

    exit_code = root_main(["list", "--full"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "run_id" in captured.out.splitlines()[0]
    assert "loss" in captured.out.splitlines()[0]
    assert "retry" in captured.out.splitlines()[0]
    assert "failure" in captured.out.splitlines()[0]
    assert "t=1" in captured.out
    assert "v=0.9" in captured.out
    assert "cuda out of memory" in captured.out
    assert "nan explosion in validation step" in captured.out


def test_unified_list_supports_shared_status_and_subfolder_filters(monkeypatch, tmp_path, capsys):
    queue_root = tmp_path / ".lisai" / "queue"
    datasets_root = tmp_path / "datasets"
    monkeypatch.setenv("LISAI_QUEUE_ROOT", str(queue_root))
    monkeypatch.setattr(unified_cli, "scan_runs", lambda: scan_runs(datasets_root))

    _write_run(
        datasets_root / "Actin" / "models" / "Upsamp" / "comp_run_00",
        run_id="01ARZ3NDEKTSV4RRFFQ69G5FAC",
        dataset="Actin",
        model_subfolder="Upsamp",
        status="completed",
    )
    now = datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc)
    queued_job = QueueJob(
        job_id="job_queued_only",
        selector="q0003",
        config=str(tmp_path / "cfg_q.yml"),
        status="queued",
        device="cuda:0",
        submitted_at=now - timedelta(minutes=8),
        updated_at=now - timedelta(minutes=8),
        resource_class="light",
        dataset="Actin",
        model_subfolder="HDN",
        run_name="queued_only",
    )
    _write_queue_job(queued_job, queue_root=queue_root)

    exit_code = root_main(["list", "--status", "completed"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "comp_run" in captured.out
    assert "job_queued_only" not in captured.out

    exit_code = root_main(["list", "--status", "queued"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "queued_only" in captured.out
    assert "comp_run" not in captured.out

    exit_code = root_main(["list", "--subfolder", "Upsamp"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "comp_run" in captured.out
    assert "queued_only" not in captured.out


def test_unified_list_orders_terminal_then_running_then_queued(monkeypatch, tmp_path, capsys):
    queue_root = tmp_path / ".lisai" / "queue"
    datasets_root = tmp_path / "datasets"
    monkeypatch.setenv("LISAI_QUEUE_ROOT", str(queue_root))
    monkeypatch.setattr(unified_cli, "scan_runs", lambda: scan_runs(datasets_root))

    # terminal run
    term_dir = datasets_root / "Gag" / "models" / "HDN" / "term_run_00"
    _write_run(
        term_dir,
        run_id="01ARZ3NDEKTSV4RRFFQ69G5FB3",
        dataset="Gag",
        model_subfolder="HDN",
        status="completed",
    )
    # running run
    running_dir = datasets_root / "Gag" / "models" / "HDN" / "active_run_00"
    _write_run(
        running_dir,
        run_id="01ARZ3NDEKTSV4RRFFQ69G5FB4",
        dataset="Gag",
        model_subfolder="HDN",
        status="running",
    )

    now = datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc)
    queued_job = QueueJob(
        job_id="job_order_queued",
        selector="q0020",
        config=str(tmp_path / "cfg_q.yml"),
        status="queued",
        device="cuda:0",
        submitted_at=now - timedelta(minutes=1),
        updated_at=now - timedelta(minutes=1),
        resource_class="light",
        dataset="Gag",
        model_subfolder="HDN",
        run_name="queued_only",
    )
    _write_queue_job(queued_job, queue_root=queue_root)

    exit_code = root_main(["list"])
    captured = capsys.readouterr()
    assert exit_code == 0
    lines = [line for line in captured.out.splitlines() if line.strip()]
    body = lines[2:]
    assert "term_run" in body[0]
    assert "active_run" in body[1]
    assert "queued_only" in body[-1]


def test_unified_list_failed_jobs_do_not_merge_into_running_linked_run(monkeypatch, tmp_path, capsys):
    queue_root = tmp_path / ".lisai" / "queue"
    datasets_root = tmp_path / "datasets"
    monkeypatch.setenv("LISAI_QUEUE_ROOT", str(queue_root))
    monkeypatch.setattr(unified_cli, "scan_runs", lambda: scan_runs(datasets_root))

    run_dir = datasets_root / "Gag_timelapses" / "models" / "Upsamp" / "CL1_Upsamp05_biggerNet_00"
    _write_run(
        run_dir,
        run_id="01ARZ3NDEKTSV4RRFFQ69G5FB5",
        dataset="Gag_timelapses",
        model_subfolder="Upsamp",
        status="running",
        last_epoch=127,
        max_epoch=200,
    )

    now = datetime(2026, 4, 8, 12, 0, tzinfo=timezone.utc)
    failed_a = QueueJob(
        job_id="job_failed_a",
        selector="q0070",
        config=str(tmp_path / "upsamp.yml"),
        status="failed",
        device="cuda:0",
        submitted_at=now - timedelta(days=5),
        updated_at=now - timedelta(days=5),
        resource_class="medium",
        dataset="Gag_timelapses",
        model_subfolder="Upsamp",
        run_name="CL1_Upsamp05_biggerNet",
        error="crashed",
        finished_at=now - timedelta(days=5),
    )
    failed_b = QueueJob(
        job_id="job_failed_b",
        selector="q0071",
        config=str(tmp_path / "continue_training.yml"),
        status="failed",
        device="cuda:0",
        submitted_at=now - timedelta(days=1),
        updated_at=now - timedelta(days=1),
        resource_class="medium",
        dataset="Gag_timelapses",
        model_subfolder="Upsamp",
        run_name="CL1_Upsamp05_biggerNet",
        error="killed",
        finished_at=now - timedelta(days=1),
    )
    _write_queue_job(failed_a, queue_root=queue_root)
    _write_queue_job(failed_b, queue_root=queue_root)

    exit_code = root_main(["list"])
    captured = capsys.readouterr()
    assert exit_code == 0
    lines = [line for line in captured.out.splitlines() if line.strip()]
    body = lines[2:]
    q0070_line = next(line for line in body if "q0070" in line)
    q0071_line = next(line for line in body if "q0071" in line)
    assert "failed" in q0070_line
    assert "failed" in q0071_line
    assert "running" not in q0070_line
    assert "running" not in q0071_line
    # Run remains visible as run-only because no running/done job owns it.
    assert any(line.split()[0] == "-" and "CL1_Upsamp05_biggerNet" in line for line in body)


def test_unified_list_done_completed_merge_without_run_duplicate(monkeypatch, tmp_path, capsys):
    queue_root = tmp_path / ".lisai" / "queue"
    datasets_root = tmp_path / "datasets"
    monkeypatch.setenv("LISAI_QUEUE_ROOT", str(queue_root))
    monkeypatch.setattr(unified_cli, "scan_runs", lambda: scan_runs(datasets_root))

    run_dir = datasets_root / "Gag" / "models" / "Upsamp" / "done_merge_00"
    metadata = _write_run(
        run_dir,
        run_id="01ARZ3NDEKTSV4RRFFQ69G5FB6",
        dataset="Gag",
        model_subfolder="Upsamp",
        status="completed",
    )
    now = datetime(2026, 4, 8, 12, 0, tzinfo=timezone.utc)
    done_job = QueueJob(
        job_id="job_done_merge",
        selector="q0072",
        config=str(tmp_path / "upsamp.yml"),
        status="done",
        device="cuda:0",
        submitted_at=now - timedelta(hours=3),
        updated_at=now - timedelta(hours=2),
        resource_class="medium",
        run_id=metadata.run_id,
        dataset="Gag",
        model_subfolder="Upsamp",
        run_name="done_merge",
        finished_at=now - timedelta(hours=2),
        exit_code=0,
    )
    _write_queue_job(done_job, queue_root=queue_root)

    exit_code = root_main(["list"])
    captured = capsys.readouterr()
    assert exit_code == 0
    body = [line for line in captured.out.splitlines()[2:] if line.strip()]
    assert sum("done_merge" in line for line in body) == 1
    assert any("q0072" in line for line in body)
    assert not any(line.split()[0] == "-" and "done_merge" in line for line in body)


def test_unified_list_done_or_blocked_hidden_when_linked_run_is_running(monkeypatch, tmp_path, capsys):
    queue_root = tmp_path / ".lisai" / "queue"
    datasets_root = tmp_path / "datasets"
    monkeypatch.setenv("LISAI_QUEUE_ROOT", str(queue_root))
    monkeypatch.setattr(unified_cli, "scan_runs", lambda: scan_runs(datasets_root))

    _write_run(
        datasets_root / "Gag" / "models" / "Upsamp" / "still_running_00",
        run_id="01ARZ3NDEKTSV4RRFFQ69G5FB7",
        dataset="Gag",
        model_subfolder="Upsamp",
        status="running",
    )
    now = datetime(2026, 4, 8, 12, 0, tzinfo=timezone.utc)
    done_job = QueueJob(
        job_id="job_done_old",
        selector="q0073",
        config=str(tmp_path / "upsamp.yml"),
        status="done",
        device="cuda:0",
        submitted_at=now - timedelta(days=1),
        updated_at=now - timedelta(days=1),
        resource_class="medium",
        dataset="Gag",
        model_subfolder="Upsamp",
        run_name="still_running",
        finished_at=now - timedelta(days=1),
        exit_code=0,
    )
    blocked_job = QueueJob(
        job_id="job_blocked_old",
        selector="q0074",
        config=str(tmp_path / "upsamp.yml"),
        status="blocked",
        device="cuda:0",
        submitted_at=now - timedelta(hours=20),
        updated_at=now - timedelta(hours=20),
        resource_class="medium",
        dataset="Gag",
        model_subfolder="Upsamp",
        run_name="still_running",
        finished_at=now - timedelta(hours=20),
        error="prelaunch_validation_failed",
    )
    _write_queue_job(done_job, queue_root=queue_root)
    _write_queue_job(blocked_job, queue_root=queue_root)

    exit_code = root_main(["list"])
    captured = capsys.readouterr()
    assert exit_code == 0
    body = [line for line in captured.out.splitlines()[2:] if line.strip()]
    assert not any("q0073" in line for line in body)
    assert not any("q0074" in line for line in body)
    assert any(line.split()[0] == "-" and "still_running" in line and "running" in line for line in body)


def test_unified_show_supports_queue_and_run_targets(monkeypatch, tmp_path, capsys):
    queue_root = tmp_path / ".lisai" / "queue"
    datasets_root = tmp_path / "datasets"
    monkeypatch.setenv("LISAI_QUEUE_ROOT", str(queue_root))
    monkeypatch.setattr(unified_cli, "scan_runs", lambda: scan_runs(datasets_root))

    run_dir = datasets_root / "Gag" / "models" / "HDN" / "show_run_00"
    metadata = _write_run(
        run_dir,
        run_id="01ARZ3NDEKTSV4RRFFQ69G5FAD",
        dataset="Gag",
        model_subfolder="HDN",
        status="running",
    )
    _write_loss(run_dir, train_loss=0.7, val_loss=0.5)

    now = datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc)
    queue_job = QueueJob(
        job_id="job_show_target",
        selector="q0004",
        config=str(tmp_path / "cfg_show.yml"),
        status="running",
        device="cuda:0",
        submitted_at=now - timedelta(minutes=6),
        updated_at=now - timedelta(minutes=1),
        resource_class="medium",
        run_id=metadata.run_id,
        dataset="Gag",
        model_subfolder="HDN",
        run_name="show_run",
        launched_at=now - timedelta(minutes=6),
    )
    _write_queue_job(queue_job, queue_root=queue_root)

    exit_code = root_main(["show", "q0004"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "kind          : queue_job" in captured.out
    assert "latest_loss   : t=0.7 v=0.5" in captured.out

    exit_code = root_main(["show", metadata.run_id])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "kind          : run" in captured.out
    assert f"run_id        : {metadata.run_id}" in captured.out
    assert "retry         : 1/1" in captured.out


def test_unified_logs_uses_canonical_queue_and_run_log_paths(monkeypatch, tmp_path, capsys):
    queue_root = tmp_path / ".lisai" / "queue"
    datasets_root = tmp_path / "datasets"
    monkeypatch.setenv("LISAI_QUEUE_ROOT", str(queue_root))
    monkeypatch.setattr(unified_cli, "scan_runs", lambda: scan_runs(datasets_root))

    run_dir = datasets_root / "Gag" / "models" / "HDN" / "log_run_00"
    metadata = _write_run(
        run_dir,
        run_id="01ARZ3NDEKTSV4RRFFQ69G5FAE",
        dataset="Gag",
        model_subfolder="HDN",
        status="running",
    )
    _write_run_log(run_dir, "run-log-line-1\nrun-log-line-2\n")

    now = datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc)
    queue_job = QueueJob(
        job_id="job_log_target",
        selector="q0005",
        config=str(tmp_path / "cfg_log.yml"),
        status="running",
        device="cuda:0",
        submitted_at=now - timedelta(minutes=9),
        updated_at=now - timedelta(minutes=1),
        resource_class="medium",
        launched_at=now - timedelta(minutes=8),
        log_path=None,
    )
    _write_queue_job(queue_job, queue_root=queue_root)
    qlog = queue_logs_dir(queue_root=queue_root) / "job_q0005_job_log_target.log"
    qlog.parent.mkdir(parents=True, exist_ok=True)
    qlog.write_text("queue-log-line-1\nqueue-log-line-2\n", encoding="utf-8")

    exit_code = root_main(["logs", "q0005", "--lines", "1"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "queue-log-line-2" in captured.out

    exit_code = root_main(["logs", metadata.run_id, "--lines", "1"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "run-log-line-2" in captured.out


def test_unified_show_reports_ambiguity_non_interactive(monkeypatch, tmp_path, capsys):
    queue_root = tmp_path / ".lisai" / "queue"
    datasets_root = tmp_path / "datasets"
    monkeypatch.setenv("LISAI_QUEUE_ROOT", str(queue_root))
    monkeypatch.setattr(unified_cli, "scan_runs", lambda: scan_runs(datasets_root))

    _write_run(
        datasets_root / "Gag" / "models" / "HDN" / "job_shared_00",
        run_id="01ARZ3NDEKTSV4RRFFQ69G5FAF",
        dataset="Gag",
        model_subfolder="HDN",
        status="running",
    )
    now = datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc)
    queue_job = QueueJob(
        job_id="job_shared_target",
        selector="q0006",
        config=str(tmp_path / "cfg_shared.yml"),
        status="queued",
        device="cuda:0",
        submitted_at=now - timedelta(minutes=3),
        updated_at=now - timedelta(minutes=3),
        resource_class="light",
        run_name="job_shared",
    )
    _write_queue_job(queue_job, queue_root=queue_root)

    exit_code = root_main(["show", "job_shared"])
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Multiple jobs/runs match this selector" in captured.out
    assert "exact qNNNN/job_id/run_id selector" in captured.err


def test_unified_show_allows_interactive_ambiguity_selection(monkeypatch, tmp_path):
    queue_root = tmp_path / ".lisai" / "queue"
    datasets_root = tmp_path / "datasets"
    monkeypatch.setenv("LISAI_QUEUE_ROOT", str(queue_root))
    monkeypatch.setattr(unified_cli, "scan_runs", lambda: scan_runs(datasets_root))

    _write_run(
        datasets_root / "Gag" / "models" / "HDN" / "job_shared_00",
        run_id="01ARZ3NDEKTSV4RRFFQ69G5FB0",
        dataset="Gag",
        model_subfolder="HDN",
        status="running",
    )
    now = datetime(2026, 3, 20, 12, 0, tzinfo=timezone.utc)
    queue_job = QueueJob(
        job_id="job_shared_target",
        selector="q0007",
        config=str(tmp_path / "cfg_shared.yml"),
        status="queued",
        device="cuda:0",
        submitted_at=now - timedelta(minutes=3),
        updated_at=now - timedelta(minutes=3),
        resource_class="light",
        run_name="job_shared",
    )
    _write_queue_job(queue_job, queue_root=queue_root)

    class _InteractiveInput(StringIO):
        def isatty(self):
            return True

    stdin = _InteractiveInput("02\n")
    stdout = StringIO()
    stderr = StringIO()
    exit_code = unified_cli.show_target(target="job_shared", stdin=stdin, stdout=stdout, stderr=stderr)
    assert exit_code == 0
    assert "kind          : run" in stdout.getvalue()
    assert stderr.getvalue() == ""
