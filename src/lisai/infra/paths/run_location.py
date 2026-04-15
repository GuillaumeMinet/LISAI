from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


_RUNS_CONTAINER_DIRNAME = "models"


@dataclass(frozen=True)
class InferredRunLocation:
    metadata_path: Path
    run_dir: Path
    dataset: str
    model_subfolder: str
    group_path: str | None
    path: str


def dataset_models_dir(dataset_dir: str | Path) -> Path:
    return Path(dataset_dir) / _RUNS_CONTAINER_DIRNAME


def iter_run_metadata_paths(
    datasets_root: str | Path,
    *,
    metadata_filename: str,
) -> Iterable[Path]:
    root = Path(datasets_root).resolve()
    for dataset_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        models_dir = dataset_models_dir(dataset_dir)
        if not models_dir.is_dir():
            continue
        for meta_path in sorted(models_dir.rglob(metadata_filename)):
            yield meta_path


def infer_run_location(
    metadata_path: str | Path,
    datasets_root: str | Path,
    *,
    metadata_filename: str,
) -> InferredRunLocation:
    meta_path = Path(metadata_path).resolve()
    root = Path(datasets_root).resolve()
    relative = meta_path.relative_to(root)
    parts = relative.parts

    if len(parts) < 5:
        raise ValueError(
            f"Run metadata path is too shallow to identify dataset/model_subfolder/run_dir: {meta_path}"
        )
    if parts[1] != _RUNS_CONTAINER_DIRNAME:
        raise ValueError(
            f"Run metadata path must live under datasets/*/{_RUNS_CONTAINER_DIRNAME}/: {meta_path}"
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
    "dataset_models_dir",
    "infer_run_location",
    "iter_run_metadata_paths",
]
