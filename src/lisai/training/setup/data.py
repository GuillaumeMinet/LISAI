from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from lisai.config import load_yaml
from lisai.data.data_prep import make_training_loaders

from .noise_model import resolve_noise_model_metadata

if TYPE_CHECKING:
    from lisai.config.models import ResolvedExperiment
    from lisai.training.runtime import TrainingRuntime

logger = logging.getLogger("lisai.prepare_data")


@dataclass
class PreparedTrainingData:
    """Typed output of training data preparation."""

    train_loader: Any
    val_loader: Any
    data_norm_prm: dict | None
    model_norm_prm: dict | None
    patch_info: Any | None



def prepare_data(
    cfg: ResolvedExperiment,
    runtime: TrainingRuntime,
) -> PreparedTrainingData:
    """
    Resolve training data configuration, normalization, and loaders.
    """
    is_lvae = cfg.model.architecture == "lvae"
    is_volumetric = cfg.model.architecture == "unet3d"

    data_norm_prm = None
    if is_lvae:
        data_norm_prm = resolve_noise_model_metadata(cfg, runtime.paths)
    if data_norm_prm is None:
        data_norm_prm = (cfg.normalization or {}).get("norm_prm")

    data_dir = runtime.paths.dataset_dir(
        dataset_name=cfg.data.dataset_name,
        data_subfolder=cfg.routing.data_subfolder,
    )

    registry = {}
    try:
        registry = load_yaml(runtime.paths.dataset_registry_path())
    except FileNotFoundError:
        logger.warning("Dataset registry not found")

    dataset_info = registry.get(cfg.data.dataset_name, None)

    data_cfg = cfg.data.resolved(
        data_dir=data_dir,
        norm_prm=data_norm_prm,
        dataset_info=dataset_info,
        volumetric=is_volumetric,
    )

    train_loader, val_loader, model_norm_prm, patch_info = make_training_loaders(
        config=data_cfg,
    )

    return PreparedTrainingData(
        train_loader=train_loader,
        val_loader=val_loader,
        data_norm_prm=data_norm_prm,
        model_norm_prm=model_norm_prm,
        patch_info=patch_info,
    )
