from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from lisai.config import settings

from .io import read_run_metadata, write_run_metadata_atomic
from .schema import SCHEMA_VERSION, RunMetadata, normalize_posix_path, utc_now


def normalize_model_subfolder(model_subfolder: str | None) -> str | None:
    if model_subfolder is None:
        return None
    text = model_subfolder.replace("\\", "/").strip().strip("/")
    if not text:
        return None
    return normalize_posix_path(text)


def group_path_from_model_subfolder(model_subfolder: str | None) -> str | None:
    normalized = normalize_model_subfolder(model_subfolder)
    if normalized is None:
        return None
    parts = normalized.split("/")
    if len(parts) <= 1:
        return None
    return "/".join(parts[1:])


def stored_run_path(run_dir: str | Path) -> str:
    path = Path(run_dir).resolve()
    data_dir = Path(settings.resolve_path(settings.project.paths.roots["data_dir"])).resolve()
    data_root = data_dir.parent

    try:
        return path.relative_to(data_root).as_posix()
    except ValueError:
        return path.as_posix()


def create_run_metadata(
    run_dir: str | Path,
    *,
    dataset: str,
    model_subfolder: str | None = None,
    max_epoch: int | None = None,
    group_path: str | None = None,
    preserve_existing: bool = False,
) -> RunMetadata:
    run_dir = Path(run_dir)
    normalized_model_subfolder = normalize_model_subfolder(model_subfolder)
    if normalized_model_subfolder is None:
        normalized_model_subfolder = run_dir.parent.name.strip() or "unknown_model_subfolder"
    group_path = group_path if group_path is not None else group_path_from_model_subfolder(normalized_model_subfolder)
    path_text = stored_run_path(run_dir)
    now = utc_now()
    existing = _read_existing_metadata(run_dir) if preserve_existing else None

    if existing is None:
        metadata = RunMetadata(
            schema_version=SCHEMA_VERSION,
            run_id=run_dir.name,
            dataset=dataset.strip(),
            model_subfolder=normalized_model_subfolder,
            status="running",
            closed_cleanly=False,
            created_at=now,
            updated_at=now,
            ended_at=None,
            last_heartbeat_at=now,
            last_epoch=None,
            max_epoch=max_epoch,
            best_val_loss=None,
            path=path_text,
            group_path=group_path,
        )
    else:
        metadata = existing.model_copy(
            update={
                "run_id": run_dir.name,
                "dataset": dataset.strip(),
                "model_subfolder": normalized_model_subfolder,
                "status": "running",
                "closed_cleanly": False,
                "updated_at": now,
                "ended_at": None,
                "last_heartbeat_at": now,
                "max_epoch": max_epoch if max_epoch is not None else existing.max_epoch,
                "path": path_text,
                "group_path": group_path,
            }
        )

    write_run_metadata_atomic(run_dir, metadata)
    return metadata


def update_run_heartbeat(run_dir: str | Path) -> RunMetadata:
    metadata = read_run_metadata(run_dir)
    now = utc_now()
    updated = metadata.model_copy(
        update={
            "status": "running",
            "closed_cleanly": False,
            "updated_at": now,
            "ended_at": None,
            "last_heartbeat_at": now,
        }
    )
    write_run_metadata_atomic(run_dir, updated)
    return updated


def update_run_progress(
    run_dir: str | Path,
    *,
    last_epoch: int | None = None,
    max_epoch: int | None = None,
    val_loss: float | None = None,
) -> RunMetadata:
    metadata = read_run_metadata(run_dir)
    now = utc_now()
    best_val_loss = metadata.best_val_loss
    if val_loss is not None:
        val_loss = float(val_loss)
        if best_val_loss is None or val_loss < best_val_loss:
            best_val_loss = val_loss

    updated = metadata.model_copy(
        update={
            "status": "running",
            "closed_cleanly": False,
            "updated_at": now,
            "ended_at": None,
            "last_heartbeat_at": now,
            "last_epoch": metadata.last_epoch if last_epoch is None else last_epoch,
            "max_epoch": metadata.max_epoch if max_epoch is None else max_epoch,
            "best_val_loss": best_val_loss,
        }
    )
    write_run_metadata_atomic(run_dir, updated)
    return updated


def finalize_run_completed(run_dir: str | Path) -> RunMetadata:
    return _finalize_run(run_dir, status="completed")


def finalize_run_stopped(run_dir: str | Path) -> RunMetadata:
    return _finalize_run(run_dir, status="stopped")


def finalize_run_failed(run_dir: str | Path) -> RunMetadata:
    return _finalize_run(run_dir, status="failed")


def _finalize_run(run_dir: str | Path, *, status: str) -> RunMetadata:
    metadata = read_run_metadata(run_dir)
    now = utc_now()
    updated = metadata.model_copy(
        update={
            "status": status,
            "closed_cleanly": True,
            "updated_at": now,
            "ended_at": now,
            "last_heartbeat_at": now,
        }
    )
    write_run_metadata_atomic(run_dir, updated)
    return updated


def _read_existing_metadata(run_dir: str | Path) -> RunMetadata | None:
    try:
        return read_run_metadata(run_dir)
    except (FileNotFoundError, OSError, ValidationError, ValueError, json.JSONDecodeError):
        return None


__all__ = [
    "create_run_metadata",
    "finalize_run_completed",
    "finalize_run_failed",
    "finalize_run_stopped",
    "group_path_from_model_subfolder",
    "normalize_model_subfolder",
    "stored_run_path",
    "update_run_heartbeat",
    "update_run_progress",
]
