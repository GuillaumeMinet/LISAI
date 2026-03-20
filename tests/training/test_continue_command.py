from __future__ import annotations

import io
from datetime import timedelta
from pathlib import Path

import lisai.cli as root_cli
import lisai.training.continue_cli as continue_cli
from lisai.config import settings
from lisai.runs.io import write_run_metadata_atomic
from lisai.runs.scanner import scan_runs
from lisai.runs.schema import RunMetadata, utc_now


class InteractiveInput(io.StringIO):
    def isatty(self) -> bool:
        return True


class NonInteractiveInput(io.StringIO):
    def isatty(self) -> bool:
        return False


def _write_metadata(
    run_dir: Path,
    *,
    dataset: str,
    model_subfolder: str,
    status: str = "running",
    closed_cleanly: bool | None = None,
    last_heartbeat_at=None,
    ended_at=None,
):
    now = utc_now()
    heartbeat = last_heartbeat_at if last_heartbeat_at is not None else now
    created_at = heartbeat - timedelta(minutes=5)
    updated_at = heartbeat

    if closed_cleanly is None:
        closed_cleanly = status != "running"
    if ended_at is None and status != "running":
        ended_at = heartbeat

    payload = {
        "schema_version": 1,
        "run_id": run_dir.name,
        "dataset": dataset,
        "model_subfolder": model_subfolder,
        "status": status,
        "closed_cleanly": closed_cleanly,
        "created_at": created_at,
        "updated_at": updated_at,
        "ended_at": ended_at,
        "last_heartbeat_at": heartbeat,
        "last_epoch": 3,
        "max_epoch": 10,
        "best_val_loss": 0.4,
        "path": f"datasets/{dataset}/models/{model_subfolder}/{run_dir.name}",
        "group_path": None if "/" not in model_subfolder else model_subfolder.split("/", 1)[1],
    }
    write_run_metadata_atomic(run_dir, RunMetadata.model_validate(payload))


def test_root_cli_continue_dispatches_unique_match_and_builds_continue_config(monkeypatch, tmp_path):
    datasets_root = tmp_path / "datasets"
    run_dir = datasets_root / "Gag_timelapses" / "models" / "Upsamp" / "CL1_Upsamp2_Mltpl05_lightweight_01"
    _write_metadata(
        run_dir,
        dataset="Gag_timelapses",
        model_subfolder="Upsamp",
        status="completed",
    )

    captured = {}
    monkeypatch.setattr(continue_cli, "scan_runs", lambda: scan_runs(datasets_root))
    monkeypatch.setattr(continue_cli, "run_training_from_config_dict", lambda cfg: captured.update({"cfg": cfg}))

    exit_code = root_cli.main([
        "continue",
        "CL1_Upsamp2_Mltpl05_lightweight_01",
        "--dataset",
        "Gag_timelapses",
        "--yes",
    ])

    assert exit_code == 0
    assert captured["cfg"] == {
        "experiment": {"mode": "continue_training"},
        "load_model": {
            "canonical_load": True,
            "dataset_name": "Gag_timelapses",
            "subfolder": "Upsamp",
            "exp_name": "CL1_Upsamp2_Mltpl05_lightweight_01",
            "load_method": "state_dict",
            "best_or_last": "last",
        },
    }


def test_continue_reports_multiple_matches_and_requests_disambiguation(monkeypatch, tmp_path):
    datasets_root = tmp_path / "datasets"
    run_id = "duplicate_run"
    _write_metadata(
        datasets_root / "Gag" / "models" / "HDN" / run_id,
        dataset="Gag",
        model_subfolder="HDN",
        status="completed",
    )
    _write_metadata(
        datasets_root / "Actin" / "models" / "Upsamp" / run_id,
        dataset="Actin",
        model_subfolder="Upsamp",
        status="completed",
    )

    stdout = io.StringIO()
    stderr = io.StringIO()
    monkeypatch.setattr(continue_cli, "scan_runs", lambda: scan_runs(datasets_root))

    exit_code = continue_cli.continue_run(
        run_id=run_id,
        stdin=NonInteractiveInput(""),
        stdout=stdout,
        stderr=stderr,
    )

    assert exit_code == 1
    assert "Multiple matching runs found" in stdout.getvalue()
    assert run_id in stdout.getvalue()
    assert "Rerun with --dataset and/or --subfolder to disambiguate." in stderr.getvalue()


def test_continue_requires_yes_when_confirmation_is_non_interactive(monkeypatch, tmp_path):
    datasets_root = tmp_path / "datasets"
    run_id = "resume_me"
    _write_metadata(
        datasets_root / "Gag" / "models" / "HDN" / run_id,
        dataset="Gag",
        model_subfolder="HDN",
        status="completed",
    )

    captured = {"called": False}
    stdout = io.StringIO()
    stderr = io.StringIO()
    monkeypatch.setattr(continue_cli, "scan_runs", lambda: scan_runs(datasets_root))
    monkeypatch.setattr(continue_cli, "run_training_from_config_dict", lambda cfg: captured.update({"called": True}))

    exit_code = continue_cli.continue_run(
        run_id=run_id,
        stdin=NonInteractiveInput(""),
        stdout=stdout,
        stderr=stderr,
    )

    assert exit_code == 1
    assert captured["called"] is False
    assert "Confirmation required. Rerun with --yes to continue non-interactively." in stderr.getvalue()


def test_continue_blocks_recently_active_runs_without_force(monkeypatch, tmp_path):
    datasets_root = tmp_path / "datasets"
    run_id = "still_running"
    now = utc_now()
    _write_metadata(
        datasets_root / "Gag" / "models" / "HDN" / run_id,
        dataset="Gag",
        model_subfolder="HDN",
        status="running",
        closed_cleanly=False,
        last_heartbeat_at=now - timedelta(minutes=2),
        ended_at=None,
    )

    captured = {"called": False}
    stderr = io.StringIO()
    monkeypatch.setattr(continue_cli, "scan_runs", lambda: scan_runs(datasets_root))
    monkeypatch.setattr(continue_cli, "run_training_from_config_dict", lambda cfg: captured.update({"called": True}))
    monkeypatch.setattr(settings.project.run_tracking, "active_heartbeat_timeout_minutes", 10)

    exit_code = continue_cli.continue_run(
        run_id=run_id,
        assume_yes=True,
        force=False,
        stdin=NonInteractiveInput(""),
        stdout=io.StringIO(),
        stderr=stderr,
        now=now,
    )

    assert exit_code == 1
    assert captured["called"] is False
    assert "Rerun with --force" in stderr.getvalue()


def test_continue_force_allows_recently_active_runs(monkeypatch, tmp_path):
    datasets_root = tmp_path / "datasets"
    run_id = "force_resume"
    now = utc_now()
    _write_metadata(
        datasets_root / "Gag" / "models" / "HDN" / run_id,
        dataset="Gag",
        model_subfolder="HDN",
        status="running",
        closed_cleanly=False,
        last_heartbeat_at=now - timedelta(minutes=2),
        ended_at=None,
    )

    captured = {}
    stderr = io.StringIO()
    monkeypatch.setattr(continue_cli, "scan_runs", lambda: scan_runs(datasets_root))
    monkeypatch.setattr(continue_cli, "run_training_from_config_dict", lambda cfg: captured.update({"cfg": cfg}))
    monkeypatch.setattr(settings.project.run_tracking, "active_heartbeat_timeout_minutes", 10)

    exit_code = continue_cli.continue_run(
        run_id=run_id,
        assume_yes=True,
        force=True,
        stdin=NonInteractiveInput(""),
        stdout=io.StringIO(),
        stderr=stderr,
        now=now,
    )

    assert exit_code == 0
    assert captured["cfg"]["load_model"]["exp_name"] == run_id
    assert "warning: forcing continuation" in stderr.getvalue()


def test_continue_allows_stale_running_runs_after_confirmation(monkeypatch, tmp_path):
    datasets_root = tmp_path / "datasets"
    run_id = "stale_run"
    now = utc_now()
    _write_metadata(
        datasets_root / "Gag" / "models" / "HDN" / run_id,
        dataset="Gag",
        model_subfolder="HDN",
        status="running",
        closed_cleanly=False,
        last_heartbeat_at=now - timedelta(minutes=30),
        ended_at=None,
    )

    captured = {}
    stderr = io.StringIO()
    monkeypatch.setattr(continue_cli, "scan_runs", lambda: scan_runs(datasets_root))
    monkeypatch.setattr(continue_cli, "run_training_from_config_dict", lambda cfg: captured.update({"cfg": cfg}))
    monkeypatch.setattr(settings.project.run_tracking, "active_heartbeat_timeout_minutes", 10)

    exit_code = continue_cli.continue_run(
        run_id=run_id,
        stdin=InteractiveInput("y\n"),
        stdout=io.StringIO(),
        stderr=stderr,
        now=now,
    )

    assert exit_code == 0
    assert captured["cfg"]["load_model"]["exp_name"] == run_id
    assert "appears stale" in stderr.getvalue()