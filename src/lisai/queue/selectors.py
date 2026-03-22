from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Callable, TypeVar

from .schema import SELECTOR_MIN_WIDTH, format_queue_selector
from .storage import ensure_queue_dirs

try:
    import fcntl  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - non-POSIX fallback
    fcntl = None


SELECTOR_STATE_SCHEMA_VERSION = 1
SELECTOR_STATE_FILENAME = "selectors.json"
SELECTOR_LOCK_FILENAME = ".selectors.lock"

_T = TypeVar("_T")


def selectors_state_path(*, queue_root: str | Path | None = None) -> Path:
    root = ensure_queue_dirs(queue_root=queue_root)
    return root / SELECTOR_STATE_FILENAME


def allocate_selector(*, queue_root: str | Path | None = None) -> str:
    def _allocate(state: dict[str, int]) -> tuple[str, dict[str, int]]:
        next_index = int(state["next_index"])
        selector = format_queue_selector(next_index, width=SELECTOR_MIN_WIDTH)
        updated = dict(state)
        updated["next_index"] = next_index + 1
        return selector, updated

    return _mutate_selector_state(_allocate, queue_root=queue_root)


def reset_selector_index(*, queue_root: str | Path | None = None) -> None:
    def _reset(_state: dict[str, int]) -> tuple[None, dict[str, int]]:
        return None, _default_selector_state()

    _mutate_selector_state(_reset, queue_root=queue_root)


def read_next_selector_index(*, queue_root: str | Path | None = None) -> int:
    state = _read_selector_state(queue_root=queue_root)
    return int(state["next_index"])


def _default_selector_state() -> dict[str, int]:
    return {
        "schema_version": SELECTOR_STATE_SCHEMA_VERSION,
        "next_index": 1,
    }


def _read_selector_state(*, queue_root: str | Path | None = None) -> dict[str, int]:
    path = selectors_state_path(queue_root=queue_root)
    if not path.exists():
        return _default_selector_state()

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        raise ValueError("Invalid selector state: expected object payload.")

    schema_version = int(payload.get("schema_version", 0))
    if schema_version != SELECTOR_STATE_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported selector state schema_version {schema_version!r}. "
            f"Expected {SELECTOR_STATE_SCHEMA_VERSION}."
        )

    next_index = int(payload.get("next_index", 0))
    if next_index <= 0:
        raise ValueError("Invalid selector state: next_index must be >= 1.")
    return {"schema_version": schema_version, "next_index": next_index}


def _write_selector_state_atomic(
    state: dict[str, int],
    *,
    queue_root: str | Path | None = None,
) -> None:
    path = selectors_state_path(queue_root=queue_root)
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
            json.dump(state, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
        _fsync_directory(path.parent)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise


def _mutate_selector_state(
    mutate: Callable[[dict[str, int]], tuple[_T, dict[str, int]]],
    *,
    queue_root: str | Path | None = None,
) -> _T:
    root = ensure_queue_dirs(queue_root=queue_root)
    lock_path = root / SELECTOR_LOCK_FILENAME
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_handle = lock_path.open("a+", encoding="utf-8")
    try:
        if fcntl is not None:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)

        state = _read_selector_state(queue_root=root)
        result, updated_state = mutate(state)
        _write_selector_state_atomic(updated_state, queue_root=root)
        return result
    finally:
        try:
            if fcntl is not None:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        finally:
            lock_handle.close()


def _fsync_directory(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


__all__ = [
    "allocate_selector",
    "read_next_selector_index",
    "reset_selector_index",
    "selectors_state_path",
]
