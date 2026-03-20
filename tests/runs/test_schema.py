from __future__ import annotations

import pytest
from pydantic import ValidationError

from lisai.runs.schema import RunMetadata


def _payload(**overrides):
    payload = {
        "schema_version": 1,
        "run_id": "HDN_Gag_KL07_01",
        "dataset": "Gag",
        "model_subfolder": "HDN",
        "status": "running",
        "closed_cleanly": False,
        "created_at": "2026-03-20T10:14:00Z",
        "updated_at": "2026-03-20T10:15:00Z",
        "ended_at": None,
        "last_heartbeat_at": "2026-03-20T10:15:00Z",
        "last_epoch": 17,
        "max_epoch": 100,
        "best_val_loss": None,
        "path": "datasets/Gag/models/HDN/HDN_Gag_KL07_01",
        "group_path": None,
    }
    payload.update(overrides)
    return payload


def test_run_metadata_accepts_running_payload():
    metadata = RunMetadata.model_validate(_payload())

    assert metadata.status == "running"
    assert metadata.ended_at is None
    assert metadata.model_subfolder == "HDN"





@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("status", "stale"),
        ("created_at", "not-a-timestamp"),
        ("extra_field", True),

        ("models_subfolder", "HDN"),
    ],
)
def test_run_metadata_rejects_invalid_values(field: str, value):
    payload = _payload()
    payload[field] = value

    with pytest.raises(ValidationError):
        RunMetadata.model_validate(payload)


def test_run_metadata_rejects_missing_required_field():
    payload = _payload()
    payload.pop("dataset")

    with pytest.raises(ValidationError):
        RunMetadata.model_validate(payload)

