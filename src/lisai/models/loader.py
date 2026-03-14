from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from lisai.infra.config import settings
from lisai.infra.paths import Paths
from lisai.runtime._old_runs_compatibility import extract_norm_prm
from lisai.runtime.inference import build_inference_spec, iter_inference_checkpoint_candidates
from lisai.runtime.spec import ModelSpec

from .registry import get_model_class
from .load_nm import load_noise_model

logger = logging.getLogger("lisai.model")


def _to_scalar(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().numpy()
    return value


def _model_norm_from_model(model) -> dict[str, Any] | None:
    if not hasattr(model, "data_mean") or not hasattr(model, "data_std"):
        return None
    return {
        "data_mean": _to_scalar(getattr(model, "data_mean", None)),
        "data_std": _to_scalar(getattr(model, "data_std", None)),
        "data_mean_gt": _to_scalar(getattr(model, "data_mean_gt", None)),
        "data_std_gt": _to_scalar(getattr(model, "data_std_gt", None)),
    }



def init_model(
    architecture: str,
    model_prm: dict,
    device: torch.device,
    *,
    model_norm_prm: dict | None = None,
    noise_model=None,
    img_shape: int | None = None,
):
    ModelClass = get_model_class(architecture)

    if architecture == "lvae":
        missing = []
        if model_norm_prm is None:
            missing.append("model_norm_prm")
        if img_shape is None:
            missing.append("img_shape")
        if noise_model is None:
            missing.append("noise_model")
        if missing:
            raise ValueError(f"LVAE initialization missing required args: {', '.join(missing)}")

        lvae_prm = dict(model_prm or {})
        lvae_prm["img_shape"] = (img_shape, img_shape)
        lvae_prm["norm_prm"] = model_norm_prm

        z_dims = lvae_prm.get("z_dims")
        n_latents = int(lvae_prm.get("num_latents", 1))
        if isinstance(z_dims, int):
            lvae_prm["z_dims"] = [z_dims] * n_latents

        model = ModelClass(device, noiseModel=noise_model, **lvae_prm)
    else:
        model = ModelClass(**(model_prm or {}))

    return model.to(device)


def _compute_img_shape(patch_size: int | None, downsamp_factor: int | None) -> int | None:
    if patch_size is None:
        return None
    ds = downsamp_factor or 1
    try:
        return int(patch_size) // int(ds)
    except Exception:
        return None


def _origin_checkpoint_path(spec: ModelSpec) -> Path:
    if spec.origin_run_dir is None:
        raise ValueError("origin_run_dir is required to load a checkpoint.")

    origin_dir = Path(spec.origin_run_dir)
    paths = Paths(settings)

    # Filename takes precedence (no need for load_method)
    if spec.checkpoint_filename:
        return paths.checkpoint_path(
            run_dir=origin_dir,
            model_name=spec.checkpoint_filename,
        )

    method = spec.checkpoint_method or "state_dict"
    selector = spec.checkpoint_selector
    epoch = spec.checkpoint_epoch
    mode = spec.mode

    if selector is None and epoch is None:
        if mode == "continue_training":
            selector = "last"
        elif mode == "retrain":
            selector = "best"
        else:
            selector = "best"

    return paths.checkpoint_path(
        run_dir=origin_dir,
        load_method=method,
        best_or_last=selector,
        epoch_number=epoch,
    )


def prepare_model_for_training(
    *,
    spec: ModelSpec,
    device: torch.device,
    model_norm_prm: dict | None = None,
    noise_model: None
):
    """
    Build (and optionally load) model based on ModelSpec.
    """
    arch = spec.architecture
    model_prm = spec.parameters or {}

    if not arch:
        raise ValueError("ModelSpec.architecture is empty")
    
    if arch == "lvae" and noise_model is None:
        raise ValueError("Need noise model to perform lvae training")

    should_load = spec.mode in {"continue_training", "retrain"}

    # full_model path
    if should_load:
        ckpt_method = spec.checkpoint_method or "state_dict"
        origin_ckpt = _origin_checkpoint_path(spec)
        logger.info(f"Loading checkpoint from: {origin_ckpt}")

        if not origin_ckpt.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {origin_ckpt}")

        if ckpt_method == "full_model":
            model = torch.load(origin_ckpt, map_location=device)
            if spec.mode == "continue_training":
                warnings.warn(
                    "continue_training with full_model load: optimizer/scheduler state handling depends on your trainer."
                )
            return model, None

    # instantiate model
    img_shape = _compute_img_shape(spec.patch_size, spec.downsamp_factor)
    model = init_model(
        architecture=arch,
        model_prm=model_prm,
        device=device,
        model_norm_prm=model_norm_prm,
        noise_model=noise_model,
        img_shape=img_shape,
    )

    state = None
    if should_load:
        origin_ckpt = _origin_checkpoint_path(spec)
        loaded = torch.load(origin_ckpt, map_location=device)

        if isinstance(loaded, dict) and "model_state_dict" in loaded:
            model.load_state_dict(loaded["model_state_dict"])
            state = loaded
        elif isinstance(loaded, dict):
            model.load_state_dict(loaded)
            state = None
        else:
            raise ValueError(f"Unsupported checkpoint type at {origin_ckpt}: {type(loaded)}")

    return model, state


def get_model_for_inference(
    model_folder: Path,
    device: torch.device,
    best_or_last: str = "best",
    epoch_number: int | None = None,
):
    """
    Load model + training config for inference/evaluation from a training run folder.

    Returns:
        (model, training_cfg, is_lvae)
    """
    spec, training_cfg, _training_cfg_raw = build_inference_spec(
        model_folder=Path(model_folder),
        best_or_last=best_or_last,
        epoch_number=epoch_number,
    )

    selected_method = None
    checkpoint_path = None
    checked_paths = []
    for method, ckpt_path in iter_inference_checkpoint_candidates(spec):
        checked_paths.append(str(ckpt_path))
        if ckpt_path.exists():
            selected_method = method
            checkpoint_path = ckpt_path
            break

    if checkpoint_path is None:
        raise FileNotFoundError(
            "Could not find a model checkpoint for inference. Checked:\n"
            + "\n".join(checked_paths)
        )

    is_lvae = spec.architecture == "lvae"

    if selected_method == "full_model":
        model = torch.load(checkpoint_path, map_location=device)
    else:
        model_prm = spec.parameters or {}
        data_prm = training_cfg.get("data_prm") or {}
        model_norm_prm = spec.model_norm_prm

        noise_model = None
        if is_lvae:
            paths = Paths(settings)
            noise_model, nm_norm_prm = load_noise_model(spec.noise_model_name, device, paths)
            if model_norm_prm is None and nm_norm_prm is not None:
                model_norm_prm = dict(nm_norm_prm)

            if model_norm_prm is None:
                norm_prm = extract_norm_prm(training_cfg, data_prm)
                if norm_prm is not None:
                    model_norm_prm = dict(norm_prm)

        img_shape = _compute_img_shape(spec.patch_size, spec.downsamp_factor)
        model = init_model(
            architecture=spec.architecture,
            model_prm=model_prm,
            device=device,
            model_norm_prm=model_norm_prm,
            noise_model=noise_model,
            img_shape=img_shape,
        )

        loaded = torch.load(checkpoint_path, map_location=device)
        if isinstance(loaded, dict) and "model_state_dict" in loaded:
            model.load_state_dict(loaded["model_state_dict"])
        elif isinstance(loaded, dict):
            model.load_state_dict(loaded)
        else:
            raise ValueError(f"Unsupported checkpoint type at {checkpoint_path}: {type(loaded)}")

    model.eval()

    merged_model_norm = training_cfg.get("model_norm_prm")
    inferred_model_norm = _model_norm_from_model(model)
    if inferred_model_norm is not None:
        if isinstance(merged_model_norm, Mapping):
            merged = dict(merged_model_norm)
            for key, value in inferred_model_norm.items():
                merged.setdefault(key, value)
            training_cfg["model_norm_prm"] = merged
        else:
            training_cfg["model_norm_prm"] = inferred_model_norm

    return model, training_cfg, is_lvae



