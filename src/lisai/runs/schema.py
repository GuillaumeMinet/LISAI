from __future__ import annotations

from datetime import datetime, timezone
from pathlib import PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator, model_validator

from .identifiers import is_valid_run_id

RUN_METADATA_FILENAME = ".lisai_run_meta.json"
SCHEMA_VERSION = 2
RUN_STATUSES = ("running", "completed", "stopped", "failed")
RunStatus = Literal["running", "completed", "stopped", "failed"]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def parse_timestamp(value: datetime | str | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        normalized = value.strip()
        if normalized.endswith("Z"):
            normalized = f"{normalized[:-1]}+00:00"
        dt = datetime.fromisoformat(normalized)
    else:
        raise TypeError(f"Unsupported timestamp type: {type(value)!r}")

    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError("Timestamp must be timezone-aware.")
    return dt.astimezone(timezone.utc)


def format_timestamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def normalize_posix_path(value: str) -> str:
    text = value.replace("\\", "/").strip()
    if not text:
        raise ValueError("Path must not be empty.")
    normalized = PurePosixPath(text).as_posix()
    if normalized == ".":
        raise ValueError("Path must not be '.'.")
    return normalized


class RunMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_version: int = Field(default=SCHEMA_VERSION)
    run_id: str
    run_name: str
    run_index: int = Field(ge=0)
    dataset: str
    model_subfolder: str

    status: RunStatus
    closed_cleanly: bool
    created_at: datetime
    updated_at: datetime
    ended_at: datetime | None
    last_heartbeat_at: datetime
    last_epoch: int | None = None
    max_epoch: int | None = None
    best_val_loss: float | None = None
    path: str
    group_path: str | None = None



    @field_validator("schema_version")
    @classmethod
    def _validate_schema_version(cls, value: int) -> int:
        if value != SCHEMA_VERSION:
            raise ValueError(f"Unsupported schema_version {value!r}. Expected {SCHEMA_VERSION}.")
        return value

    @field_validator("run_id", mode="before")
    @classmethod
    def _normalize_run_id(cls, value: str) -> str:
        if not isinstance(value, str):
            raise TypeError("run_id must be a string.")
        text = value.strip().upper()
        if not text:
            raise ValueError("run_id must not be empty.")
        if not is_valid_run_id(text):
            raise ValueError("run_id must be a ULID-like 26-character identifier.")
        return text

    @field_validator("run_name", "dataset", "model_subfolder")
    @classmethod
    def _validate_required_text(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("Value must not be empty.")
        return text

    @field_validator("created_at", "updated_at", "last_heartbeat_at", mode="before")
    @classmethod
    def _parse_required_timestamps(cls, value: datetime | str) -> datetime:
        parsed = parse_timestamp(value)
        if parsed is None:
            raise ValueError("Timestamp must not be null.")
        return parsed

    @field_validator("ended_at", mode="before")
    @classmethod
    def _parse_optional_timestamp(cls, value: datetime | str | None) -> datetime | None:
        return parse_timestamp(value)

    @field_validator("path", mode="before")
    @classmethod
    def _normalize_path(cls, value: str) -> str:
        if not isinstance(value, str):
            raise TypeError("Path must be a string.")
        return normalize_posix_path(value)

    @field_validator("model_subfolder", "group_path", mode="before")
    @classmethod
    def _normalize_optional_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise TypeError("Path value must be a string.")
        normalized = normalize_posix_path(value)
        return None if normalized in {"", "."} else normalized

    @field_validator("last_epoch")
    @classmethod
    def _validate_last_epoch(cls, value: int | None) -> int | None:
        if value is not None and value < 0:
            raise ValueError("last_epoch must be >= 0.")
        return value

    @field_validator("max_epoch")
    @classmethod
    def _validate_max_epoch(cls, value: int | None) -> int | None:
        if value is not None and value < 0:
            raise ValueError("max_epoch must be >= 0.")
        return value

    @model_validator(mode="after")
    def _validate_consistency(self):
        if self.updated_at < self.created_at:
            raise ValueError("updated_at must be >= created_at.")
        if self.last_heartbeat_at < self.created_at:
            raise ValueError("last_heartbeat_at must be >= created_at.")
        if self.ended_at is not None and self.ended_at < self.created_at:
            raise ValueError("ended_at must be >= created_at.")

        if self.status == "running":
            if self.ended_at is not None:
                raise ValueError("ended_at must be null while status is running.")
            if self.closed_cleanly:
                raise ValueError("closed_cleanly must be false while status is running.")
        else:
            if self.ended_at is None:
                raise ValueError("ended_at must be set for terminal statuses.")
            if not self.closed_cleanly:
                raise ValueError("closed_cleanly must be true for terminal statuses.")

        if self.last_epoch is not None and self.max_epoch is not None and self.last_epoch > self.max_epoch:
            raise ValueError("last_epoch must be <= max_epoch.")

        parts = [part for part in self.model_subfolder.split("/") if part]
        if not parts:
            raise ValueError("model_subfolder must not be empty.")
        expected_group = "/".join(parts[1:]) or None
        if self.group_path != expected_group:
            raise ValueError("group_path must match model_subfolder.")

        return self

    @field_serializer("created_at", "updated_at", "last_heartbeat_at", when_used="json")
    def _serialize_required_timestamps(self, value: datetime) -> str:
        return format_timestamp(value)

    @field_serializer("ended_at", when_used="json")
    def _serialize_optional_timestamp(self, value: datetime | None) -> str | None:
        if value is None:
            return None
        return format_timestamp(value)


__all__ = [
    "RUN_METADATA_FILENAME",
    "RUN_STATUSES",
    "SCHEMA_VERSION",
    "RunMetadata",
    "RunStatus",
    "format_timestamp",
    "normalize_posix_path",
    "parse_timestamp",
    "utc_now",
]
