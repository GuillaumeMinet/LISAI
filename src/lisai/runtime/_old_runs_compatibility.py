from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from lisai.infra.config import load_yaml, settings
from lisai.infra.paths import Paths


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def normalize_architecture(name: str | None) -> str | None:
    if name is None:
        return None
    key = str(name).strip().lower().replace("-", "_")
    aliases = {
        "unetrcan": "unet_rcan",
        "unet_rcan": "unet_rcan",
        "lvae": "lvae",
        "unet": "unet",
        "unet3d": "unet3d",
        "rcan": "rcan",
    }
    return aliases.get(key, key)


def load_training_cfg_from_run(model_folder: Path) -> dict[str, Any]:
    model_folder = Path(model_folder)
    paths = Paths(settings)
    canonical_cfg = paths.cfg_train_path(run_dir=model_folder)

    candidates = [
        canonical_cfg,
        model_folder / "config_train.yml",
        model_folder / "config_train.yaml",
        model_folder / "config_train.json",
    ]

    for cfg_path in candidates:
        if not cfg_path.exists():
            continue
        if cfg_path.suffix.lower() == ".json":
            with open(cfg_path, "r") as f:
                cfg = json.load(f)
        else:
            cfg = load_yaml(cfg_path)
        if not isinstance(cfg, dict):
            raise ValueError(f"Training config must be a mapping, found: {type(cfg)} at {cfg_path}")
        return cfg

    raise FileNotFoundError(f"Could not find a training config in run folder: {model_folder}")


def extract_data_prm(training_cfg: Mapping[str, Any]) -> dict[str, Any]:
    data_prm = _as_dict(training_cfg.get("data_prm"))
    if not data_prm:
        data_prm = _as_dict(training_cfg.get("data"))

    if data_prm.get("input") is None and isinstance(data_prm.get("inp"), str):
        data_prm["input"] = data_prm["inp"]
    if data_prm.get("target") is None and data_prm.get("gt") is not None:
        data_prm["target"] = data_prm["gt"]
    return data_prm


def extract_model_prm(training_cfg: Mapping[str, Any]) -> dict[str, Any]:
    model_prm = _as_dict(training_cfg.get("model_prm"))
    if model_prm:
        return model_prm
    model_section = _as_dict(training_cfg.get("model"))
    return _as_dict(model_section.get("parameters"))


def extract_model_architecture(training_cfg: Mapping[str, Any]) -> str | None:
    arch = training_cfg.get("model_architecture")
    if arch is None:
        arch = _as_dict(training_cfg.get("model")).get("architecture")
    return normalize_architecture(arch)


def extract_norm_prm(training_cfg: Mapping[str, Any], data_prm: Mapping[str, Any]) -> dict[str, Any] | None:
    normalization = _as_dict(training_cfg.get("normalization"))
    norm_prm = normalization.get("norm_prm")
    if isinstance(norm_prm, Mapping):
        return dict(norm_prm)
    if isinstance(data_prm.get("norm_prm"), Mapping):
        return dict(data_prm["norm_prm"])
    return None


def extract_model_norm_prm(training_cfg: Mapping[str, Any], data_prm: Mapping[str, Any]) -> dict[str, Any] | None:
    candidates = [
        training_cfg.get("model_norm_prm"),
        _as_dict(training_cfg.get("data")).get("model_norm_prm"),
        data_prm.get("model_norm_prm"),
    ]
    for item in candidates:
        if isinstance(item, Mapping):
            return dict(item)
    return None


def extract_noise_model_name(training_cfg: Mapping[str, Any]) -> str | None:
    noise_model = training_cfg.get("noise_model")
    if isinstance(noise_model, Mapping):
        return noise_model.get("name")
    if isinstance(noise_model, str):
        return noise_model
    return None


def extract_saving_cfg(training_cfg: Mapping[str, Any]) -> dict[str, Any]:
    saving_cfg = _as_dict(training_cfg.get("saving_prm"))
    if not saving_cfg:
        saving_cfg = _as_dict(training_cfg.get("saving"))
    return saving_cfg


def preferred_load_method(training_cfg: Mapping[str, Any]) -> str:
    saving_cfg = extract_saving_cfg(training_cfg)
    if saving_cfg.get("state_dict") is True:
        return "state_dict"
    if saving_cfg.get("state_dict") is False and saving_cfg.get("entire_model") is True:
        return "full_model"
    if saving_cfg.get("full_model") is True:
        return "full_model"
    return "state_dict"


def extract_patch_size_and_downsamp_factor(data_prm: Mapping[str, Any]) -> tuple[int | None, int]:
    patch_size = data_prm.get("patch_size")
    if patch_size is None:
        patch_size = data_prm.get("val_patch_size")

    downsamp_factor = 1
    downsampling = data_prm.get("downsampling")
    if isinstance(downsampling, Mapping):
        downsamp_factor = int(downsampling.get("downsamp_factor") or 1)

    legacy_downsampling = data_prm.get("downsamp_prm")
    if downsamp_factor == 1 and isinstance(legacy_downsampling, Mapping):
        downsamp_factor = int(legacy_downsampling.get("downsamp_factor") or 1)

    return int(patch_size) if patch_size is not None else None, int(downsamp_factor)


def normalize_training_cfg_for_inference(training_cfg: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(training_cfg)
    data_prm = extract_data_prm(training_cfg)
    model_prm = extract_model_prm(training_cfg)
    model_architecture = extract_model_architecture(training_cfg)
    normalization = _as_dict(training_cfg.get("normalization"))

    if normalization.get("norm_prm") is None and isinstance(data_prm.get("norm_prm"), Mapping):
        normalization["norm_prm"] = dict(data_prm["norm_prm"])

    model_norm_prm = extract_model_norm_prm(training_cfg, data_prm)

    normalized["data_prm"] = data_prm
    normalized["model_prm"] = model_prm
    normalized["model_architecture"] = model_architecture
    normalized["normalization"] = normalization
    if model_norm_prm is not None:
        normalized["model_norm_prm"] = model_norm_prm
    return normalized

