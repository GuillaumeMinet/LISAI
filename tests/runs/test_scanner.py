from __future__ import annotations

import json

from lisai.runs.io import write_run_metadata_atomic
from lisai.runs.scanner import scan_runs
from lisai.runs.schema import RunMetadata


def _write_metadata(run_dir, *, dataset, model_subfolder, group_path, path, status="running"):
    payload = {
        "schema_version": 1,
        "run_id": run_dir.name,
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


def test_scan_runs_discovers_nested_run_directories(tmp_path):
    datasets_root = tmp_path / "datasets"
    run_a = datasets_root / "Gag" / "models" / "HDN" / "HDN_Gag_KL07_01"
    run_b = datasets_root / "Gag" / "models" / "HDN" / "ablationA" / "HDN_Gag_KL07_02"
    run_c = datasets_root / "Gag" / "models" / "Upsamp" / "2026_03" / "test1" / "Upsamp_Gag_CL5"

    _write_metadata(
        run_a,
        dataset="Gag",
        model_subfolder="HDN",
        group_path=None,
        path="datasets/Gag/models/HDN/HDN_Gag_KL07_01",
    )
    _write_metadata(
        run_b,
        dataset="Gag",
        model_subfolder="HDN/ablationA",
        group_path="ablationA",
        path="datasets/Gag/models/HDN/ablationA/HDN_Gag_KL07_02",
    )
    _write_metadata(
        run_c,
        dataset="Gag",
        model_subfolder="Upsamp/2026_03/test1",
        group_path="2026_03/test1",
        path="datasets/Gag/models/Upsamp/2026_03/test1/Upsamp_Gag_CL5",
    )

    results = scan_runs(datasets_root)
    by_id = {run.metadata.run_id: run for run in results.runs}

    assert len(results.runs) == 3
    assert results.invalid == ()
    assert by_id["HDN_Gag_KL07_01"].model_subfolder == "HDN"
    assert by_id["HDN_Gag_KL07_02"].group_path == "ablationA"
    assert by_id["Upsamp_Gag_CL5"].model_subfolder == "Upsamp/2026_03/test1"


def test_scan_runs_skips_invalid_metadata_files(tmp_path):
    datasets_root = tmp_path / "datasets"
    valid_run = datasets_root / "Gag" / "models" / "HDN" / "valid_run"
    invalid_json_run = datasets_root / "Gag" / "models" / "HDN" / "broken_json"
    invalid_schema_run = datasets_root / "Gag" / "models" / "HDN" / "broken_schema"

    _write_metadata(
        valid_run,
        dataset="Gag",
        model_subfolder="HDN",
        group_path=None,
        path="datasets/Gag/models/HDN/valid_run",
    )
    (invalid_json_run / ".lisai_run_meta.json").parent.mkdir(parents=True, exist_ok=True)
    (invalid_json_run / ".lisai_run_meta.json").write_text("{bad json", encoding="utf-8")
    (invalid_schema_run / ".lisai_run_meta.json").parent.mkdir(parents=True, exist_ok=True)
    (invalid_schema_run / ".lisai_run_meta.json").write_text(
        json.dumps({"schema_version": 1, "status": "wrong"}),
        encoding="utf-8",
    )

    results = scan_runs(datasets_root)
    kinds = {item.kind for item in results.invalid}

    assert len(results.runs) == 1
    assert "json_parse_error" in kinds
    assert "schema_validation_error" in kinds


def test_scan_runs_reports_path_mismatches_without_crashing(tmp_path):
    datasets_root = tmp_path / "datasets"
    run_dir = datasets_root / "Gag" / "models" / "HDN" / "wrong_path"

    _write_metadata(
        run_dir,
        dataset="Gag",
        model_subfolder="HDN",
        group_path=None,
        path="datasets/Gag/models/HDN/not_the_real_run_dir",
    )

    results = scan_runs(datasets_root)

    assert results.runs == ()
    assert len(results.invalid) == 1
    assert results.invalid[0].kind == "path_mismatch"
