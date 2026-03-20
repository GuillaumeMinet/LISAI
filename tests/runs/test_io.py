from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from lisai.runs.io import metadata_path, read_run_metadata, write_run_metadata_atomic
from lisai.runs.schema import RunMetadata


def _payload(**overrides):
    payload = {
        "schema_version": 2,
        "run_id": "01ARZ3NDEKTSV4RRFFQ69G5FAV",
        "run_name": "run",
        "run_index": 1,
        "dataset": "Gag",
        "model_subfolder": "HDN",
        "status": "running",
        "closed_cleanly": False,
        "created_at": "2026-03-20T10:14:00Z",
        "updated_at": "2026-03-20T10:15:00Z",
        "ended_at": None,
        "last_heartbeat_at": "2026-03-20T10:15:00Z",
        "last_epoch": 2,
        "max_epoch": 10,
        "best_val_loss": 0.5,
        "path": "datasets/Gag/models/HDN/run_01",
        "group_path": None,
    }
    payload.update(overrides)
    return payload


def test_write_and_read_run_metadata_round_trip(tmp_path):
    run_dir = tmp_path / "run_01"
    metadata = RunMetadata.model_validate(_payload())

    out_path = write_run_metadata_atomic(run_dir, metadata)
    loaded = read_run_metadata(run_dir)

    assert out_path == metadata_path(run_dir)
    assert loaded.run_id == metadata.run_id
    assert loaded.best_val_loss == metadata.best_val_loss
    assert loaded.model_subfolder == metadata.model_subfolder
    assert sorted(path.name for path in run_dir.iterdir()) == [".lisai_run_meta.json"]


def test_read_run_metadata_raises_on_invalid_json(tmp_path):
    run_dir = tmp_path / "run_02"
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata_path(run_dir).write_text("{bad json", encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        read_run_metadata(run_dir)


def test_read_run_metadata_raises_on_schema_validation_error(tmp_path):
    run_dir = tmp_path / "run_03"
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata_path(run_dir).write_text(json.dumps({"schema_version": 2}), encoding="utf-8")

    with pytest.raises(ValidationError):
        read_run_metadata(run_dir)
