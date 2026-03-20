from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Mapping

from .schema import RUN_METADATA_FILENAME, RunMetadata


def metadata_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / RUN_METADATA_FILENAME


def _coerce_metadata_path(path_or_run_dir: str | Path) -> Path:
    path = Path(path_or_run_dir)
    if path.name == RUN_METADATA_FILENAME:
        return path
    return metadata_path(path)


def read_run_metadata(path_or_run_dir: str | Path) -> RunMetadata:
    path = _coerce_metadata_path(path_or_run_dir)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return RunMetadata.model_validate(payload)


def write_run_metadata_atomic(
    path_or_run_dir: str | Path,
    metadata: RunMetadata | Mapping[str, object],
) -> Path:
    path = _coerce_metadata_path(path_or_run_dir)
    model = metadata if isinstance(metadata, RunMetadata) else RunMetadata.model_validate(metadata)

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
        text=True,
    )
    tmp_path = Path(tmp_name)

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(model.model_dump(mode="json"), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

        os.replace(tmp_path, path)
        _fsync_directory(path.parent)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise

    return path


def _fsync_directory(path: Path):
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return

    try:
        os.fsync(fd)
    finally:
        os.close(fd)


__all__ = ["metadata_path", "read_run_metadata", "write_run_metadata_atomic"]
