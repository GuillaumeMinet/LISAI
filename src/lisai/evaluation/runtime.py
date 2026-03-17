from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from lisai.config import settings
from lisai.infra.paths import Paths
from lisai.models import load_noise_model
from lisai.models.loader import init_model

from ._training_run_compat import (
    extract_data_prm,
    extract_model_architecture,
    extract_model_norm_prm,
    extract_model_prm,
    extract_noise_model_name,
    extract_norm_prm,
    extract_patch_size_and_downsamp_factor,
    load_training_cfg_from_run,
    normalize_training_cfg_for_inference,
    preferred_load_method,
)
from .context import resolve_context_length, resolve_tiling_size, resolve_upsampling_factor


@dataclass
class InferenceRuntime:
    """Loaded evaluation runtime for one trained run."""

    paths: Paths
    device: torch.device
    run_dir: Path
    model: Any
    training_cfg: dict[str, Any]
    training_cfg_raw: dict[str, Any]
    data_prm: dict[str, Any]
    architecture: str
    is_lvae: bool
    data_norm_prm: dict[str, Any] | None
    model_norm_prm: dict[str, Any] | None
    noise_model_name: str | None
    patch_size: int | None
    downsamp_factor: int
    tiling_size: int | None
    upsampling_factor: int
    context_length: int | None
    checkpoint_path: Path
    load_method: str



def _default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")



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



def _compute_img_shape(patch_size: int | None, downsamp_factor: int | None) -> int | None:
    if patch_size is None:
        return None
    ds = downsamp_factor or 1
    try:
        return int(patch_size) // int(ds)
    except Exception:
        return None



def _iter_checkpoint_candidates(
    *,
    run_dir: Path,
    checkpoint_method: str,
    checkpoint_selector: str,
    checkpoint_epoch: int | None,
    paths: Paths,
):
    methods: list[str] = []
    for method in (checkpoint_method, "state_dict", "full_model"):
        if method and method not in methods:
            methods.append(method)

    for method in methods:
        kwargs: dict[str, Any] = {"run_dir": run_dir, "load_method": method}
        if checkpoint_epoch is not None:
            kwargs["epoch_number"] = checkpoint_epoch
        else:
            kwargs["best_or_last"] = checkpoint_selector
        yield method, paths.checkpoint_path(**kwargs)



def _resolve_checkpoint_path(
    *,
    run_dir: Path,
    checkpoint_method: str,
    checkpoint_selector: str,
    checkpoint_epoch: int | None,
    paths: Paths,
) -> tuple[str, Path]:
    checked_paths = []
    for method, checkpoint_path in _iter_checkpoint_candidates(
        run_dir=run_dir,
        checkpoint_method=checkpoint_method,
        checkpoint_selector=checkpoint_selector,
        checkpoint_epoch=checkpoint_epoch,
        paths=paths,
    ):
        checked_paths.append(str(checkpoint_path))
        if checkpoint_path.exists():
            return method, checkpoint_path

    raise FileNotFoundError(
        "Could not find a model checkpoint for inference. Checked:\n" + "\n".join(checked_paths)
    )



def _load_model_from_run(
    *,
    run_dir: Path,
    device: torch.device,
    training_cfg: dict[str, Any],
    training_cfg_raw: dict[str, Any],
    checkpoint_selector: str,
    checkpoint_epoch: int | None,
    paths: Paths,
) -> tuple[str, Path, Any]:
    architecture = extract_model_architecture(training_cfg)
    if not architecture:
        raise ValueError("Could not resolve model architecture from training config.")

    data_prm = extract_data_prm(training_cfg)
    model_prm = extract_model_prm(training_cfg)
    patch_size, downsamp_factor = extract_patch_size_and_downsamp_factor(data_prm)
    noise_model_name = extract_noise_model_name(training_cfg_raw)
    checkpoint_method = preferred_load_method(training_cfg_raw)

    selected_method, checkpoint_path = _resolve_checkpoint_path(
        run_dir=run_dir,
        checkpoint_method=checkpoint_method,
        checkpoint_selector=checkpoint_selector,
        checkpoint_epoch=checkpoint_epoch,
        paths=paths,
    )

    is_lvae = architecture == "lvae"

    if selected_method == "full_model":
        model = torch.load(checkpoint_path, map_location=device)
    else:
        model_norm_prm = extract_model_norm_prm(training_cfg, data_prm)
        noise_model = None

        if is_lvae:
            noise_model, nm_norm_prm = load_noise_model(noise_model_name, device, paths)
            if model_norm_prm is None and nm_norm_prm is not None:
                model_norm_prm = dict(nm_norm_prm)

            if model_norm_prm is None:
                norm_prm = extract_norm_prm(training_cfg, data_prm)
                if norm_prm is not None:
                    model_norm_prm = dict(norm_prm)

        model = init_model(
            architecture=architecture,
            model_prm=model_prm,
            device=device,
            model_norm_prm=model_norm_prm,
            noise_model=noise_model,
            img_shape=_compute_img_shape(patch_size, downsamp_factor),
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

    return selected_method, checkpoint_path, model



def initialize_runtime(
    *,
    model_folder: Path,
    device: torch.device | str | None = None,
    best_or_last: str = "best",
    epoch_number: int | None = None,
    tiling_size: int | None = None,
) -> InferenceRuntime:
    run_dir = Path(model_folder)
    paths = Paths(settings)
    resolved_device = _default_device() if device is None else torch.device(device)

    training_cfg_raw = load_training_cfg_from_run(run_dir)
    training_cfg = normalize_training_cfg_for_inference(training_cfg_raw)

    architecture = extract_model_architecture(training_cfg)
    if not architecture:
        raise ValueError("Could not resolve model architecture from training config.")

    data_prm = extract_data_prm(training_cfg)
    patch_size, downsamp_factor = extract_patch_size_and_downsamp_factor(data_prm)
    noise_model_name = extract_noise_model_name(training_cfg_raw)

    load_method, checkpoint_path, model = _load_model_from_run(
        run_dir=run_dir,
        device=resolved_device,
        training_cfg=training_cfg,
        training_cfg_raw=training_cfg_raw,
        checkpoint_selector=best_or_last,
        checkpoint_epoch=epoch_number,
        paths=paths,
    )

    return InferenceRuntime(
        paths=paths,
        device=resolved_device,
        run_dir=run_dir,
        model=model,
        training_cfg=training_cfg,
        training_cfg_raw=training_cfg_raw,
        data_prm=dict(data_prm),
        architecture=architecture,
        is_lvae=(architecture == "lvae"),
        data_norm_prm=extract_norm_prm(training_cfg, data_prm),
        model_norm_prm=training_cfg.get("model_norm_prm"),
        noise_model_name=noise_model_name,
        patch_size=patch_size,
        downsamp_factor=downsamp_factor,
        tiling_size=resolve_tiling_size(training_cfg, tiling_size),
        upsampling_factor=resolve_upsampling_factor(training_cfg),
        context_length=resolve_context_length(training_cfg),
        checkpoint_path=checkpoint_path,
        load_method=load_method,
    )
