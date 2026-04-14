from __future__ import annotations

from pathlib import PurePosixPath


def _normalize_posix_path(value: str) -> str:
    text = value.replace("\\", "/").strip()
    if not text:
        raise ValueError("Path must not be empty.")
    normalized = PurePosixPath(text).as_posix()
    if normalized == ".":
        raise ValueError("Path must not be '.'.")
    return normalized


def normalize_model_subfolder(model_subfolder: str | None) -> str | None:
    if model_subfolder is None:
        return None
    text = model_subfolder.replace("\\", "/").strip().strip("/")
    if not text:
        return None
    return _normalize_posix_path(text)


def group_path_from_model_subfolder(model_subfolder: str | None) -> str | None:
    normalized = normalize_model_subfolder(model_subfolder)
    if normalized is None:
        return None
    parts = normalized.split("/")
    if len(parts) <= 1:
        return None
    return "/".join(parts[1:])


__all__ = [
    "group_path_from_model_subfolder",
    "normalize_model_subfolder",
]
