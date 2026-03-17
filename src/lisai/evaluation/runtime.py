"""Live evaluation runtime construction.

This module owns the runtime-side boundary of evaluation. It takes a
`SavedTrainingRun`, resolves the checkpoint to load, materializes the model on a
chosen device, and returns the small `InferenceRuntime` object used by the
entrypoints.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import torch

from lisai.config import settings
from lisai.infra.paths import Paths
from lisai.models import load_noise_model
from lisai.models.loader import init_model

from .saved_run import CheckpointMethod, SavedTrainingRun



@dataclass
class InferenceRuntime:
    """Live resources needed to run inference for one evaluation call."""

    model: Any
    device: torch.device
    checkpoint_path: Path
    load_method: CheckpointMethod
    tiling_size: int | None
    resolved_epoch: int | None



def _default_device() -> torch.device:
    """Pick CUDA when available, otherwise fall back to CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")



def _compute_img_shape(patch_size: int | None, downsamp_factor: int) -> int | None:
    """Convert saved patch metadata into the image size expected by LVAE models."""
    if patch_size is None:
        return None
    return int(patch_size) // max(1, int(downsamp_factor))



def _iter_checkpoint_candidates(
    saved_run: SavedTrainingRun,
    *,
    best_or_last: str,
    epoch_number: int | None,
    paths: Paths,
):
    """Yield candidate checkpoint paths allowed by the saved run configuration."""
    for method in saved_run.checkpoint_methods:
        kwargs: dict[str, Any] = {"run_dir": saved_run.run_dir, "load_method": method}
        if epoch_number is not None:
            kwargs["epoch_number"] = epoch_number
        else:
            kwargs["best_or_last"] = best_or_last
        yield method, paths.checkpoint_path(**kwargs)



def _resolve_checkpoint_path(
    saved_run: SavedTrainingRun,
    *,
    best_or_last: str,
    epoch_number: int | None,
    paths: Paths,
) -> tuple[CheckpointMethod, Path]:
    """Find the first existing checkpoint matching the requested selector."""
    checked_paths: list[str] = []
    for method, checkpoint_path in _iter_checkpoint_candidates(
        saved_run,
        best_or_last=best_or_last,
        epoch_number=epoch_number,
        paths=paths,
    ):
        checked_paths.append(str(checkpoint_path))
        if checkpoint_path.exists():
            return method, checkpoint_path

    raise FileNotFoundError(
        "Could not find a model checkpoint for inference. Checked:\n" + "\n".join(checked_paths)
    )



def _epoch_from_checkpoint_path(checkpoint_path: Path) -> int | None:
    match = re.search(r"model_epoch_(\d+)", checkpoint_path.name)
    if match is None:
        return None
    return int(match.group(1))



def _load_state_dict_model(
    saved_run: SavedTrainingRun,
    checkpoint_path: Path,
    device: torch.device,
    paths: Paths,
) -> tuple[Any, int | None]:
    """Instantiate the model structure and load weights from a state-dict checkpoint."""
    model_norm_prm = dict(saved_run.model_norm_prm) if saved_run.model_norm_prm is not None else None
    noise_model = None

    if saved_run.is_lvae:
        if not saved_run.noise_model_name:
            raise ValueError("Saved LVAE run is missing noise_model.name.")
        noise_model, nm_norm_prm = load_noise_model(saved_run.noise_model_name, device, paths)
        if model_norm_prm is None and nm_norm_prm is not None:
            model_norm_prm = dict(nm_norm_prm)
        if model_norm_prm is None and saved_run.data_norm_prm is not None:
            model_norm_prm = dict(saved_run.data_norm_prm)

    model = init_model(
        architecture=saved_run.model_architecture,
        model_prm=saved_run.model_parameters,
        device=device,
        model_norm_prm=model_norm_prm,
        noise_model=noise_model,
        img_shape=_compute_img_shape(saved_run.patch_size, saved_run.downsamp_factor),
    )

    loaded = torch.load(checkpoint_path, map_location=device)
    resolved_epoch = _epoch_from_checkpoint_path(checkpoint_path)
    if isinstance(loaded, dict):
        epoch = loaded.get("epoch")
        if epoch is not None:
            try:
                resolved_epoch = int(epoch)
            except (TypeError, ValueError):
                pass

    if isinstance(loaded, dict) and "model_state_dict" in loaded:
        model.load_state_dict(loaded["model_state_dict"])
    elif isinstance(loaded, dict):
        model.load_state_dict(loaded)
    else:
        raise ValueError(f"Unsupported checkpoint type at {checkpoint_path}: {type(loaded)}")

    model.eval()
    return model, resolved_epoch



def initialize_runtime(
    *,
    saved_run: SavedTrainingRun,
    device: torch.device | str | None = None,
    best_or_last: str = "best",
    epoch_number: int | None = None,
    tiling_size: int | None = None,
) -> InferenceRuntime:
    """Load the requested checkpoint and build the live inference runtime."""
    paths = Paths(settings)
    resolved_device = _default_device() if device is None else torch.device(device)
    load_method, checkpoint_path = _resolve_checkpoint_path(
        saved_run,
        best_or_last=best_or_last,
        epoch_number=epoch_number,
        paths=paths,
    )

    if load_method == "full_model":
        model = torch.load(checkpoint_path, map_location=resolved_device)
        model.eval()
        resolved_epoch = _epoch_from_checkpoint_path(checkpoint_path)
    else:
        model, resolved_epoch = _load_state_dict_model(saved_run, checkpoint_path, resolved_device, paths)

    effective_tiling_size = tiling_size if tiling_size is not None else saved_run.default_tiling_size
    return InferenceRuntime(
        model=model,
        device=resolved_device,
        checkpoint_path=checkpoint_path,
        load_method=load_method,
        tiling_size=effective_tiling_size,
        resolved_epoch=resolved_epoch,
    )



__all__ = ["InferenceRuntime", "initialize_runtime"]
