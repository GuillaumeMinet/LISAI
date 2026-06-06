from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from .paths import Paths

@dataclass(frozen=True)
class InferredRunLocation:
    metadata_path: Path
    run_dir: Path
    dataset: str
    model_subfolder: str
    group_path: str | None
    path: str


def iter_run_metadata_paths(
    datasets_root: str | Path,
    *,
    metadata_filename: str,
    paths: Paths,
) -> Iterable[Path]:
    root = Path(datasets_root).resolve()
    for dataset_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        runs_dir = paths.dataset_runs_dir_from_dataset_dir(dataset_dir)
        if not runs_dir.is_dir():
            continue
        yield from sorted(runs_dir.rglob(metadata_filename))


def infer_run_location(
    metadata_path: str | Path,
    datasets_root: str | Path,
    *,
    metadata_filename: str,
    run_container_dirname: str,
) -> InferredRunLocation:
    container = str(run_container_dirname).strip().strip("/\\")
    if not container:
        raise ValueError("run_container_dirname must not be empty.")
    meta_path = Path(metadata_path).resolve()
    root = Path(datasets_root).resolve()
    relative = meta_path.relative_to(root)
    parts = relative.parts

    if len(parts) < 5:
        raise ValueError(
            f"Run metadata path is too shallow to identify dataset/model_subfolder/run_dir: {meta_path}"
        )
    if parts[1] != container:
        raise ValueError(
            f"Run metadata path must live under datasets/*/{container}/: {meta_path}"
        )
    if parts[-1] != metadata_filename:
        raise ValueError(f"Unexpected metadata filename: {meta_path.name}")

    dataset = parts[0]
    grouping_parts = parts[3:-2]
    model_subfolder = "/".join(parts[2:-2])
    group_path = "/".join(grouping_parts) or None
    run_dir = meta_path.parent
    derived_path = (Path(root.name) / Path(*parts[:-1])).as_posix()

    return InferredRunLocation(
        metadata_path=meta_path,
        run_dir=run_dir,
        dataset=dataset,
        model_subfolder=model_subfolder,
        group_path=group_path,
        path=derived_path,
    )


__all__ = [
    "InferredRunLocation",
    "infer_run_location",
    "iter_run_metadata_paths",
]
