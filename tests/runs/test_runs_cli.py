from __future__ import annotations

from datetime import datetime, timezone

from lisai.infra.fs.run_naming import parse_run_dir_name
import lisai.runs.cli as runs_cli
import lisai.runs.listing as runs_listing
from lisai.cli import main as root_main
from lisai.config import settings
from lisai.runs.io import write_run_metadata_atomic
from lisai.runs.scanner import scan_runs
from lisai.runs.schema import RunMetadata


def _write_metadata(run_dir, *, dataset, model_subfolder, group_path, path, status="running"):
    run_name, run_index = parse_run_dir_name(run_dir.name)
    payload = {
        "schema_version": 2,
        "run_id": "01ARZ3NDEKTSV4RRFFQ69G5FAV",
        "run_name": run_name,
        "run_index": run_index,
        "dataset": dataset,
        "model_subfolder": model_subfolder,
        "status": status,
        "closed_cleanly": status != "running",
        "created_at": "2026-03-20T10:14:00Z",
        "updated_at": "2026-03-20T10:15:00Z",
        "ended_at": None if status == "running" else "2026-03-20T10:20:00Z",
        "last_heartbeat_at": "2026-03-20T10:15:00Z",
        "last_epoch": 3,
        "max_epoch": 10,
        "best_val_loss": 0.4,
        "path": path,
        "group_path": group_path,
    }
    write_run_metadata_atomic(run_dir, RunMetadata.model_validate(payload))


def test_runs_list_uses_filters_and_warns_on_invalid_files(monkeypatch, tmp_path, capsys):
    datasets_root = tmp_path / "datasets"
    run_a = datasets_root / "Gag" / "models" / "HDN" / "run_a"
    run_b = datasets_root / "Actin" / "models" / "Upsamp" / "run_b"
    invalid_run = datasets_root / "Gag" / "models" / "HDN" / "broken"

    _write_metadata(
        run_a,
        dataset="Gag",
        model_subfolder="HDN",
        group_path=None,
        path="datasets/Gag/models/HDN/run_a",
        status="running",
    )
    _write_metadata(
        run_b,
        dataset="Actin",
        model_subfolder="Upsamp",
        group_path=None,
        path="datasets/Actin/models/Upsamp/run_b",
        status="completed",
    )
    (invalid_run / ".lisai_run_meta.json").parent.mkdir(parents=True, exist_ok=True)
    (invalid_run / ".lisai_run_meta.json").write_text("{bad json", encoding="utf-8")

    monkeypatch.setattr(runs_cli, "scan_runs", lambda: scan_runs(datasets_root))

    exit_code = root_main(["runs", "list", "--dataset", "Gag", "--status", "running"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "dataset" in captured.out
    assert "path_consistent" in captured.out
    assert "run_a" in captured.out
    assert "run_b" not in captured.out
    assert "warning: skipped invalid run metadata" in captured.err


def test_runs_list_model_subfolder_filter_works(monkeypatch, tmp_path, capsys):
    datasets_root = tmp_path / "datasets"
    run_a = datasets_root / "Gag" / "models" / "HDN" / "run_a"
    run_b = datasets_root / "Gag" / "models" / "Upsamp" / "run_b"

    _write_metadata(
        run_a,
        dataset="Gag",
        model_subfolder="HDN",
        group_path=None,
        path="datasets/Gag/models/HDN/run_a",
        status="running",
    )
    _write_metadata(
        run_b,
        dataset="Gag",
        model_subfolder="Upsamp",
        group_path=None,
        path="datasets/Gag/models/Upsamp/run_b",
        status="completed",
    )

    monkeypatch.setattr(runs_cli, "scan_runs", lambda: scan_runs(datasets_root))

    exit_code = root_main(["runs", "list", "--model-subfolder", "Upsamp"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "run_b" in captured.out
    assert "run_a" not in captured.out


def test_runs_list_subfolder_alias_works(monkeypatch, tmp_path, capsys):
    datasets_root = tmp_path / "datasets"
    run_a = datasets_root / "Gag" / "models" / "HDN" / "run_a"
    run_b = datasets_root / "Gag" / "models" / "Upsamp" / "run_b"

    _write_metadata(
        run_a,
        dataset="Gag",
        model_subfolder="HDN",
        group_path=None,
        path="datasets/Gag/models/HDN/run_a",
        status="running",
    )
    _write_metadata(
        run_b,
        dataset="Gag",
        model_subfolder="Upsamp",
        group_path=None,
        path="datasets/Gag/models/Upsamp/run_b",
        status="completed",
    )

    monkeypatch.setattr(runs_cli, "scan_runs", lambda: scan_runs(datasets_root))

    exit_code = root_main(["runs", "list", "--subfolder", "HDN"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "run_a" in captured.out
    assert "run_b" not in captured.out


def test_runs_list_marks_old_running_heartbeats_as_stale(monkeypatch, tmp_path, capsys):
    datasets_root = tmp_path / "datasets"
    run_a = datasets_root / "Gag" / "models" / "HDN" / "run_a"

    _write_metadata(
        run_a,
        dataset="Gag",
        model_subfolder="HDN",
        group_path=None,
        path="datasets/Gag/models/HDN/run_a",
        status="running",
    )

    monkeypatch.setattr(runs_cli, "scan_runs", lambda: scan_runs(datasets_root))
    monkeypatch.setattr(runs_listing, "utc_now", lambda: datetime(2026, 3, 20, 11, 0, tzinfo=timezone.utc))
    monkeypatch.setattr(settings.project.run_tracking, "active_heartbeat_timeout_minutes", 10)

    exit_code = root_main(["runs", "list", "--dataset", "Gag", "--status", "running"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "run_a" in captured.out
    assert "stale?" in captured.out


def test_runs_list_footer_notes_path_inconsistency_without_stderr_warning(monkeypatch, tmp_path, capsys):
    datasets_root = tmp_path / "datasets"
    run_dir = datasets_root / "Gag" / "models" / "HDN" / "run_a_00"

    _write_metadata(
        run_dir,
        dataset="Gag",
        model_subfolder="HDN",
        group_path=None,
        path="datasets/Gag/models/HDN/not_the_real_folder",
        status="completed",
    )

    monkeypatch.setattr(runs_cli, "scan_runs", lambda: scan_runs(datasets_root))

    exit_code = root_main(["runs", "list", "--dataset", "Gag"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "path_consistent" in captured.out
    assert "false" in captured.out
    assert "inconsistent path metadata" in captured.out
    assert "path_mismatch" not in captured.err
