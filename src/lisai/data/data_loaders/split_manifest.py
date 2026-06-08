from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from lisai.config.models.training import DataSection


SPLIT_MANIFEST_FILENAME = "split_manifest.json"
SPLIT_NAMES = ("train", "val", "test")


def split_manifest_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / SPLIT_MANIFEST_FILENAME


def resolve_split_manifest_path(paths: Any, run_dir: str | Path) -> Path:
    path_factory = getattr(paths, "split_manifest_path", None)
    if callable(path_factory):
        return path_factory(run_dir=run_dir)
    return split_manifest_path(run_dir)


def read_split_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    with manifest_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_split_manifest(path: str | Path, manifest: dict[str, Any]) -> Path:
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest_path


def input_root(config: DataSection) -> Path:
    if config.data_dir is None:
        raise ValueError("`data_dir` must be provided to build an unprepared split manifest.")
    return Path(config.data_dir) / (config.input or "")


def collect_input_files(config: DataSection) -> list[Path]:
    root = input_root(config)
    files: list[Path] = []
    seen: set[Path] = set()
    for image_filter in config.filters:
        for file_path in sorted(root.glob(f"*{image_filter}")):
            resolved = file_path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            files.append(file_path)
    return files


def make_unprepared_split_manifest(
    config: DataSection,
    *,
    data_format: str,
    inp_files: Iterable[Path] | None = None,
) -> dict[str, Any]:
    files = list(inp_files) if inp_files is not None else collect_input_files(config)
    if not files:
        raise FileNotFoundError(f"No input files found in {input_root(config)} with filters={config.filters}.")

    ratios = config.split_manifest.ratios
    counts = _split_counts(len(files), {"train": ratios.train, "val": ratios.val, "test": ratios.test})

    rng = np.random.default_rng(config.split_manifest.seed)
    shuffled = [files[int(idx)] for idx in rng.permutation(len(files))]

    splits: dict[str, list[dict[str, str | None]]] = {}
    start = 0
    root = input_root(config)
    for split_name in SPLIT_NAMES:
        count = counts[split_name]
        split_files = shuffled[start : start + count]
        splits[split_name] = [
            {"input": _relative_path(file_path, root), "target": None}
            for file_path in split_files
        ]
        start += count

    timelapse_prm = None
    if config.timelapse_prm is not None:
        timelapse_prm = config.timelapse_prm.model_dump(mode="json", exclude_none=False)

    return {
        "version": 1,
        "kind": "lisai_unprepared_file_split",
        "dataset_name": config.dataset_name,
        "data_format": data_format,
        "input": config.input or "",
        "target": None,
        "paired": False,
        "filters": list(config.filters),
        "ratios": ratios.model_dump(mode="json"),
        "seed": int(config.split_manifest.seed),
        "timelapse_prm": timelapse_prm,
        "splits": splits,
    }


def files_for_manifest_split(config: DataSection, manifest: dict[str, Any], split: str) -> list[Path]:
    splits = manifest.get("splits")
    if not isinstance(splits, dict) or split not in splits:
        raise ValueError(f"Split manifest does not contain split '{split}'.")

    root = input_root(config)
    entries = splits[split]
    if not isinstance(entries, list):
        raise ValueError(f"Split manifest entry for split '{split}' must be a list.")

    files: list[Path] = []
    for entry in entries:
        if not isinstance(entry, dict) or not entry.get("input"):
            raise ValueError(f"Invalid split manifest entry in split '{split}': {entry!r}")
        files.append(root / str(entry["input"]))
    return files


def manifest_split_entries(manifest: dict[str, Any], split: str) -> list[dict[str, Any]]:
    splits = manifest.get("splits")
    if not isinstance(splits, dict) or split not in splits:
        raise ValueError(f"Split manifest does not contain split '{split}'.")
    entries = splits[split]
    if not isinstance(entries, list):
        raise ValueError(f"Split manifest entry for split '{split}' must be a list.")
    return entries


def _relative_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name


def _split_counts(n_files: int, ratios: dict[str, float]) -> dict[str, int]:
    positive_splits = [name for name in SPLIT_NAMES if ratios[name] > 0]
    if n_files < len(positive_splits):
        raise ValueError(
            f"Need at least {len(positive_splits)} files for non-empty splits {positive_splits}, got {n_files}."
        )

    raw = {name: ratios[name] * n_files for name in SPLIT_NAMES}
    counts = {name: int(np.floor(raw[name])) for name in SPLIT_NAMES}

    for name in positive_splits:
        if counts[name] == 0:
            counts[name] = 1

    while sum(counts.values()) < n_files:
        candidates = sorted(
            SPLIT_NAMES,
            key=lambda name: (raw[name] - counts[name], ratios[name]),
            reverse=True,
        )
        counts[candidates[0]] += 1

    while sum(counts.values()) > n_files:
        candidates = sorted(
            (
                name
                for name in SPLIT_NAMES
                if counts[name] > (1 if ratios[name] > 0 else 0)
            ),
            key=lambda name: (raw[name] - counts[name], ratios[name]),
        )
        if not candidates:
            raise ValueError(f"Could not derive split counts for {n_files} files and ratios={ratios}.")
        counts[candidates[0]] -= 1

    return counts
