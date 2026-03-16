from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from lisai.defaults import DEFAULT_TILING_SIZE
from lisai.config import load_yaml, settings
from lisai.infra.paths import Paths
from lisai.runtime._old_runs_compatibility import (
    extract_data_prm,
    extract_model_architecture,
    extract_model_prm,
)


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def get_model_folder(*, dataset_name: str, subfolder: str, exp_name: str) -> Path:
    paths = Paths(settings)
    return paths.run_dir(dataset_name=dataset_name, models_subfolder=subfolder, exp_name=exp_name)


def resolve_tiling_size(training_cfg: Mapping[str, Any], user_tiling_size: int | None) -> int | None:
    if user_tiling_size is not None:
        return user_tiling_size

    architecture = extract_model_architecture(training_cfg)
    if architecture is None:
        return None

    candidates = [architecture, architecture.replace("_", "")]
    for key in candidates:
        if key in DEFAULT_TILING_SIZE:
            return DEFAULT_TILING_SIZE[key]
    return None


def resolve_upsampling_factor(training_cfg: Mapping[str, Any]) -> int:
    model_prm = extract_model_prm(training_cfg)
    for key in ("upsamp", "upsampling_factor"):
        value = model_prm.get(key)
        if value is not None:
            return int(value)
    return 1


def resolve_context_length(training_cfg: Mapping[str, Any]) -> int | None:
    data_prm = extract_data_prm(training_cfg)
    timelapse_prm = _as_dict(data_prm.get("timelapse_prm"))
    value = timelapse_prm.get("context_length")
    return int(value) if value is not None else None


def resolve_data_dir(training_cfg: Mapping[str, Any], data_prm: Mapping[str, Any]) -> Path | None:
    for key in ("data_dir", "full_data_path"):
        value = data_prm.get(key)
        if value:
            return Path(value)

    canonical_load = data_prm.get("canonical_load")
    if canonical_load is False:
        return None

    data_cfg = _as_dict(training_cfg.get("data"))
    routing_cfg = _as_dict(training_cfg.get("routing"))
    dataset_name = data_prm.get("dataset_name") or data_cfg.get("dataset_name")
    if not dataset_name:
        return None

    subfolder = data_prm.get("subfolder")
    if subfolder is None:
        subfolder = routing_cfg.get("data_subfolder", "")

    paths = Paths(settings)
    return paths.dataset_dir(dataset_name=dataset_name, data_subfolder=subfolder or "")


def resolve_dataset_info(dataset_name: str | None) -> dict[str, Any] | None:
    if not dataset_name:
        return None

    paths = Paths(settings)
    try:
        registry = load_yaml(paths.dataset_registry_path())
    except FileNotFoundError:
        return None

    info = registry.get(dataset_name)
    if isinstance(info, Mapping):
        return dict(info)
    return None
