from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def _normalize_filters(filters: Sequence[str]) -> set[str]:
    return {f".{ext.lower().lstrip('.')}" for ext in filters}


def list_image_files(folder: Path, filters: Sequence[str]) -> list[Path]:
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")
    if not folder.is_dir():
        raise NotADirectoryError(f"Expected a directory, got: {folder}")

    allowed = _normalize_filters(filters)
    return sorted(
        p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in allowed
    )


def ensure_output_dir(base_dir: Path, *, overwrite: bool, max_suffix: int = 10) -> Path:
    base_dir = Path(base_dir)

    if base_dir.exists():
        if overwrite:
            base_dir.mkdir(parents=True, exist_ok=True)
            return base_dir

        for i in range(max_suffix):
            candidate = Path(f"{base_dir}_{i}")
            if not candidate.exists():
                candidate.mkdir(parents=True, exist_ok=False)
                return candidate

        raise RuntimeError(
            f"Could not find a free output directory for '{base_dir}' "
            f"after trying suffixes _0 to _{max_suffix - 1}."
        )

    base_dir.mkdir(parents=True, exist_ok=False)
    return base_dir


def save_histogram(path: Path, histogram: np.ndarray) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, histogram)
    return path


def save_norm_prm(path: Path, norm_prm: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(norm_prm, f, indent=4)
    return path


def save_text(path: Path, content: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def save_gmm_parameters(model: Any, path: Path) -> Path:
    """
    Persist current GaussianMixtureNoiseModel parameters in the canonical
    NPZ layout expected by model loading code.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    trained_weight = model.weight.detach().cpu().numpy()
    min_signal = model.min_signal.detach().cpu().numpy()
    max_signal = model.max_signal.detach().cpu().numpy()
    min_sigma = np.asarray(model.min_sigma)

    np.savez(
        path,
        trained_weight=trained_weight,
        min_signal=min_signal,
        max_signal=max_signal,
        min_sigma=min_sigma,
    )
    return path
