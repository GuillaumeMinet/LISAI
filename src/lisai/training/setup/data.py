from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import TYPE_CHECKING

from lisai.config import load_yaml
from lisai.data.data_prep import make_training_loaders

if TYPE_CHECKING:
    from lisai.config.models import ResolvedExperiment
    from lisai.training.runtime import TrainingRuntime

logger = logging.getLogger("lisai.prepare_data")


def prepare_data(
    cfg: ResolvedExperiment,
    runtime: TrainingRuntime,
    *,
    data_norm_prm: dict | None = None,
):
    """
    Resolves data paths, handles volumetric logic, creates loaders.
    cfg is ResolvedExperiment (Pydantic).
    """
    is_volumetric = cfg.model.architecture == "unet3d"

    norm_prm = data_norm_prm
    if norm_prm is None:
        norm_prm = (cfg.normalization or {}).get("norm_prm")

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
        norm_prm=norm_prm,
        dataset_info=dataset_info,
        volumetric=is_volumetric,
    )

    train_loader, val_loader, model_norm_prm, patch_info = make_training_loaders(
        config=data_cfg,
    )

    loaders = SimpleNamespace(train=train_loader, val=val_loader)
    meta = SimpleNamespace(model_norm_prm=model_norm_prm, patch_info=patch_info)

    return loaders, meta
