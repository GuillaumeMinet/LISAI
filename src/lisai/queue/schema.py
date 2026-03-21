from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator, model_validator

from lisai.runs.identifiers import is_valid_run_id
from lisai.runs.schema import TrainingSignature, format_timestamp, parse_timestamp

QUEUE_SCHEMA_VERSION = 1
JOB_STATUSES = ("queued", "running", "done", "failed")
RESOURCE_CLASSES = ("light", "medium", "heavy")

JobStatus = Literal["queued", "running", "done", "failed"]
ResourceClass = Literal["light", "medium", "heavy"]


class QueueJob(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(default=QUEUE_SCHEMA_VERSION)
    job_id: str
    config: str
    status: JobStatus
    device: str
    submitted_at: datetime
    updated_at: datetime
    resource_class: ResourceClass
    run_id: str | None = None
    dataset: str | None = None
    model_subfolder: str | None = None
    run_name: str | None = None
    training_signature: TrainingSignature | None = None
    launched_at: datetime | None = None
    finished_at: datetime | None = None
    pid: int | None = Field(default=None, ge=1)
    exit_code: int | None = None
    log_path: str | None = None
    error: str | None = None

    @field_validator("schema_version")
    @classmethod
    def _validate_schema_version(cls, value: int) -> int:
        if value != QUEUE_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported queue schema_version {value!r}. Expected {QUEUE_SCHEMA_VERSION}."
            )
        return value

    @field_validator("job_id", "config", "device")
    @classmethod
    def _validate_text_fields(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("Value must not be empty.")
        return text

    @field_validator("dataset", "model_subfolder", "run_name", "log_path", "error")
    @classmethod
    def _validate_optional_text_fields(cls, value: str | None) -> str | None:
        if value is None:
            return None
        text = value.strip()
        return None if not text else text

    @field_validator("run_id", mode="before")
    @classmethod
    def _normalize_run_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise TypeError("run_id must be a string.")
        normalized = value.strip().upper()
        if not normalized:
            return None
        if not is_valid_run_id(normalized):
            raise ValueError("run_id must be a ULID-like 26-character identifier.")
        return normalized

    @field_validator(
        "submitted_at",
        "updated_at",
        "launched_at",
        "finished_at",
        mode="before",
    )
    @classmethod
    def _parse_timestamps(cls, value: datetime | str | None):
        return parse_timestamp(value)

    @model_validator(mode="after")
    def _validate_status_consistency(self):
        if self.submitted_at is None:
            raise ValueError("submitted_at must be set.")
        if self.updated_at is None:
            raise ValueError("updated_at must be set.")
        if self.updated_at < self.submitted_at:
            raise ValueError("updated_at must be >= submitted_at.")
        if self.launched_at is not None and self.launched_at < self.submitted_at:
            raise ValueError("launched_at must be >= submitted_at.")
        if self.finished_at is not None:
            reference = self.launched_at if self.launched_at is not None else self.submitted_at
            if self.finished_at < reference:
                raise ValueError("finished_at must be >= launched_at/submitted_at.")
        if self.status == "queued":
            if self.finished_at is not None:
                raise ValueError("queued jobs cannot have finished_at.")
        if self.status == "running":
            if self.launched_at is None:
                raise ValueError("running jobs must have launched_at.")
            if self.finished_at is not None:
                raise ValueError("running jobs cannot have finished_at.")
        if self.status in {"done", "failed"}:
            if self.finished_at is None:
                raise ValueError("terminal jobs must have finished_at.")
        return self

    @field_serializer(
        "submitted_at",
        "updated_at",
        "launched_at",
        "finished_at",
        when_used="json",
    )
    def _serialize_timestamps(self, value: datetime | None) -> str | None:
        if value is None:
            return None
        return format_timestamp(value)


__all__ = [
    "JOB_STATUSES",
    "JobStatus",
    "QUEUE_SCHEMA_VERSION",
    "QueueJob",
    "RESOURCE_CLASSES",
    "ResourceClass",
]
