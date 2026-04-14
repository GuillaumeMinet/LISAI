from __future__ import annotations

from lisai.runs.io import read_run_metadata, write_run_metadata_atomic
from lisai.runs.lifecycle import update_run_attempt_state, update_run_runtime_details
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
        "last_epoch": 2,
        "max_epoch": 10,
        "best_val_loss": 0.5,
        "path": f"datasets/Gag/models/HDN/{run_dir.name}",
        "group_path": None,
    }
    payload.update(overrides)
    return payload


def test_update_run_runtime_details_sets_signature_and_peak(tmp_path):
    run_dir = tmp_path / "demo_00"
    write_run_metadata_atomic(run_dir, RunMetadata.model_validate(_payload(run_dir)))

    update_run_runtime_details(
        run_dir,
        training_signature={
            "architecture": "unet",
            "batch_size": 16,
            "patch_size": 96,
        },
        peak_gpu_mem_mb=5120,
        total_training_time_sec=120.0,
        training_time_per_epoch_sec=12.0,
    )
    metadata = read_run_metadata(run_dir)

    assert metadata.training_signature is not None
    assert metadata.training_signature.architecture == "unet"
    assert metadata.training_signature.batch_size == 16
    assert metadata.training_signature.patch_size == 96
    assert metadata.runtime_stats is not None
    assert metadata.runtime_stats.peak_gpu_mem_mb == 5120
    assert metadata.runtime_stats.total_training_time_sec == 120.0
    assert metadata.runtime_stats.training_time_per_epoch_sec == 12.0
    assert metadata.status == "running"


def test_update_run_runtime_details_keeps_max_peak(tmp_path):
    run_dir = tmp_path / "demo_00"
    write_run_metadata_atomic(
        run_dir,
        RunMetadata.model_validate(
            _payload(
                run_dir,
                runtime_stats={
                    "peak_gpu_mem_mb": 4000,
                },
            )
        ),
    )

    update_run_runtime_details(run_dir, peak_gpu_mem_mb=3500)
    metadata = read_run_metadata(run_dir)
    assert metadata.runtime_stats is not None
    assert metadata.runtime_stats.peak_gpu_mem_mb == 4000


def test_update_run_runtime_details_preserves_legacy_peak_only_stats(tmp_path):
    run_dir = tmp_path / "demo_00"
    write_run_metadata_atomic(
        run_dir,
        RunMetadata.model_validate(
            _payload(
                run_dir,
                runtime_stats={
                    "peak_gpu_mem_mb": 4096,
                },
            )
        ),
    )

    update_run_runtime_details(
        run_dir,
        training_signature={
            "architecture": "unet_rcan",
            "train_batch_size": 8,
            "train_patch_size": 96,
            "val_patch_size": 128,
            "input_channels": 1,
            "upsampling_factor": 2,
        },
    )
    metadata = read_run_metadata(run_dir)

    assert metadata.training_signature is not None
    assert metadata.training_signature.architecture == "unet_rcan"
    assert metadata.runtime_stats is not None
    assert metadata.runtime_stats.peak_gpu_mem_mb == 4096
    assert metadata.runtime_stats.total_training_time_sec is None
    assert metadata.runtime_stats.training_time_per_epoch_sec is None


def test_update_run_attempt_state_updates_retry_and_pause_status(tmp_path):
    run_dir = tmp_path / "demo_00"
    write_run_metadata_atomic(run_dir, RunMetadata.model_validate(_payload(run_dir)))

    update_run_attempt_state(
        run_dir,
        status="paused",
        retry_attempt=2,
        max_retry_attempts=3,
        failure_reason="manual_pause",
    )
    metadata = read_run_metadata(run_dir)

    assert metadata.status == "paused"
    assert metadata.closed_cleanly is False
    assert metadata.ended_at is None
    assert metadata.retry_attempt == 2
    assert metadata.max_retry_attempts == 3
    assert metadata.failure_reason == "manual_pause"
