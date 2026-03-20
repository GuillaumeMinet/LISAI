from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from pydantic import ValidationError

from lisai.config import settings

from .io import read_run_metadata
from .schema import RUN_METADATA_FILENAME, RunMetadata, normalize_posix_path


@dataclass(frozen=True)
class InferredRunLocation:
    metadata_path: Path
    run_dir: Path
    dataset: str
    model_subfolder: str
    group_path: str | None
    path: str




@dataclass(frozen=True)
class DiscoveredRun:
    metadata: RunMetadata
    metadata_path: Path
    run_dir: Path
    dataset: str
    model_subfolder: str
    group_path: str | None
    path: str



    @property
    def last_seen(self) -> datetime:
        return self.metadata.last_heartbeat_at


@dataclass(frozen=True)
class InvalidRunMetadata:
    metadata_path: Path
    kind: str
    message: str


@dataclass(frozen=True)
class ScanResults:
    runs: tuple[DiscoveredRun, ...]
    invalid: tuple[InvalidRunMetadata, ...]


def default_datasets_root() -> Path:
    return Path(settings.resolve_path(settings.project.paths.roots["data_dir"])).resolve()


def scan_runs(datasets_root: str | Path | None = None) -> ScanResults:
    root = default_datasets_root() if datasets_root is None else Path(datasets_root)
    root = root.resolve()
    if not root.exists():
        return ScanResults(runs=(), invalid=())

    runs: list[DiscoveredRun] = []
    invalid: list[InvalidRunMetadata] = []

    for dataset_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        models_dir = dataset_dir / "models"
        if not models_dir.is_dir():
            continue

        for meta_path in sorted(models_dir.rglob(RUN_METADATA_FILENAME)):
            try:
                inferred = infer_run_location(meta_path, root)
                metadata = read_run_metadata(meta_path)
                mismatches = metadata_path_mismatches(metadata, inferred)
                if mismatches:
                    invalid.append(
                        InvalidRunMetadata(
                            metadata_path=meta_path,
                            kind="path_mismatch",
                            message="; ".join(mismatches),
                        )
                    )
                    continue

                runs.append(
                    DiscoveredRun(
                        metadata=metadata,
                        metadata_path=meta_path,
                        run_dir=inferred.run_dir,
                        dataset=inferred.dataset,
                        model_subfolder=inferred.model_subfolder,
                        group_path=inferred.group_path,
                        path=inferred.path,
                    )
                )
            except json.JSONDecodeError as exc:
                invalid.append(
                    InvalidRunMetadata(
                        metadata_path=meta_path,
                        kind="json_parse_error",
                        message=str(exc),
                    )
                )
            except ValidationError as exc:
                invalid.append(
                    InvalidRunMetadata(
                        metadata_path=meta_path,
                        kind="schema_validation_error",
                        message=str(exc),
                    )
                )
            except (OSError, ValueError) as exc:
                invalid.append(
                    InvalidRunMetadata(
                        metadata_path=meta_path,
                        kind="scan_error",
                        message=str(exc),
                    )
                )

    runs.sort(key=lambda run: run.last_seen, reverse=True)
    return ScanResults(runs=tuple(runs), invalid=tuple(invalid))


def infer_run_location(metadata_path: str | Path, datasets_root: str | Path) -> InferredRunLocation:
    meta_path = Path(metadata_path).resolve()
    root = Path(datasets_root).resolve()
    relative = meta_path.relative_to(root)
    parts = relative.parts

    if len(parts) < 5:
        raise ValueError(
            f"Run metadata path is too shallow to identify dataset/model_subfolder/run_dir: {meta_path}"
        )
    if parts[1] != "models":
        raise ValueError(f"Run metadata path must live under datasets/*/models/: {meta_path}")
    if parts[-1] != RUN_METADATA_FILENAME:
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


def metadata_path_mismatches(metadata: RunMetadata, inferred: InferredRunLocation) -> list[str]:
    mismatches: list[str] = []
    if metadata.run_id != inferred.run_dir.name:
        mismatches.append(
            f"run_id mismatch: metadata={metadata.run_id!r}, filesystem={inferred.run_dir.name!r}"
        )
    if metadata.dataset != inferred.dataset:
        mismatches.append(
            f"dataset mismatch: metadata={metadata.dataset!r}, filesystem={inferred.dataset!r}"
        )

    normalized_model_subfolder = normalize_posix_path(metadata.model_subfolder)
    if normalized_model_subfolder != inferred.model_subfolder:
        mismatches.append(
            "model_subfolder mismatch: "
            f"metadata={normalized_model_subfolder!r}, filesystem={inferred.model_subfolder!r}"
        )

    if metadata.group_path is not None:
        normalized_group_path = normalize_posix_path(metadata.group_path)
        if normalized_group_path != inferred.group_path:
            mismatches.append(
                f"group_path mismatch: metadata={normalized_group_path!r}, filesystem={inferred.group_path!r}"
            )

    normalized_path = normalize_posix_path(metadata.path)
    if normalized_path != inferred.path:
        mismatches.append(f"path mismatch: metadata={normalized_path!r}, filesystem={inferred.path!r}")

    return mismatches


__all__ = [
    "DiscoveredRun",
    "InferredRunLocation",
    "InvalidRunMetadata",
    "ScanResults",
    "default_datasets_root",
    "infer_run_location",
    "metadata_path_mismatches",
    "scan_runs",
]
