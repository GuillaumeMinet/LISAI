from __future__ import annotations

import pytest

from lisai.runs.io import read_run_metadata, write_run_metadata_atomic
from lisai.runs.lifecycle import update_run_progress
from lisai.runs.schema import RunMetadata


def _payload(run_dir, **overrides):
    payload = {
        "schema_version": 2,
        "run_id": "01ARZ3NDEKTSV4RRFFQ69G5FAV",
        "run_name": "demo",
        "run_index": 0,
        "dataset": "Gag",
        "model_subfolder": "HDN",
        "status": "running",
        "closed_cleanly": False,
        "created_at": "2026-03-20T10:14:00Z",
        "updated_at": "2026-03-20T10:15:00Z",
        "ended_at": None,
        "last_heartbeat_at": "2026-03-20T10:15:00Z",
        "last_epoch": 0,
        "max_epoch": 10,
        "best_val_loss": 0.5,
        "path": f"datasets/Gag/models/HDN/{run_dir.name}",
        "group_path": None,
    }
    payload.update(overrides)
    return payload


def test_update_run_progress_updates_live_runtime_stats_rolling_window(tmp_path):
    run_dir = tmp_path / "demo_00"
    write_run_metadata_atomic(run_dir, RunMetadata.model_validate(_payload(run_dir)))

    update_run_progress(run_dir, last_epoch=1, epoch_duration_s=12.0)
    update_run_progress(run_dir, last_epoch=2, epoch_duration_s=20.0)
    update_run_progress(run_dir, last_epoch=3, epoch_duration_s=30.0)
    update_run_progress(run_dir, last_epoch=4, epoch_duration_s=40.0)

    metadata = read_run_metadata(run_dir)
    assert metadata.live_runtime_stats is not None
    assert metadata.live_runtime_stats.last_epoch_duration_s == pytest.approx(40.0)
    assert metadata.live_runtime_stats.recent_epoch_durations_s == pytest.approx([20.0, 30.0, 40.0])
    assert metadata.live_runtime_stats.median_epoch_duration_s == pytest.approx(30.0)


def test_update_run_progress_rejects_negative_epoch_duration(tmp_path):
    run_dir = tmp_path / "demo_00"
    write_run_metadata_atomic(run_dir, RunMetadata.model_validate(_payload(run_dir)))

    with pytest.raises(ValueError, match="epoch_duration_s must be >= 0"):
        update_run_progress(run_dir, last_epoch=1, epoch_duration_s=-1.0)


def test_update_run_progress_rejects_non_numeric_epoch_duration(tmp_path):
    run_dir = tmp_path / "demo_00"
    write_run_metadata_atomic(run_dir, RunMetadata.model_validate(_payload(run_dir)))

    with pytest.raises(ValueError):
        update_run_progress(run_dir, last_epoch=1, epoch_duration_s="not-a-number")


def test_update_run_progress_keeps_live_runtime_stats_when_duration_is_missing(tmp_path):
    run_dir = tmp_path / "demo_00"
    write_run_metadata_atomic(
        run_dir,
        RunMetadata.model_validate(
            _payload(
                run_dir,
                live_runtime_stats={
                    "last_epoch_duration_s": 22.0,
                    "recent_epoch_durations_s": [20.0, 22.0],
                },
            )
        ),
    )

    update_run_progress(run_dir, last_epoch=2, val_loss=0.4)
    metadata = read_run_metadata(run_dir)

    assert metadata.live_runtime_stats is not None
    assert metadata.live_runtime_stats.last_epoch_duration_s == pytest.approx(22.0)
    assert metadata.live_runtime_stats.recent_epoch_durations_s == pytest.approx([20.0, 22.0])
    assert metadata.live_runtime_stats.median_epoch_duration_s == pytest.approx(21.0)
