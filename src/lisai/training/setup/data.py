from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import TYPE_CHECKING

from lisai.data.data_prep import make_training_loaders
from lisai.infra.config import load_yaml

if TYPE_CHECKING:
    from lisai.infra.config.schema import ResolvedExperiment

    from .context import TrainingContext

logger = logging.getLogger("lisai.prepare_data")

def prepare_data(cfg: ResolvedExperiment, ctx: TrainingContext):
    """
    Resolves data paths, handles volumetric logic, creates loaders.
    cfg is ResolvedExperiment (Pydantic).
    nn_norm_prm: noise model normalization param
    """
    # normalization params
    norm_prm = (cfg.normalization or {}).get("norm_prm")
    
    # resolve path using routing
    data_dir = ctx.paths.dataset_dir(
        dataset_name=cfg.data.dataset_name,
        data_subfolder=cfg.routing.data_subfolder,
    )

    # get dataset info from registry
    registry = {}
    try:
        registry = load_yaml(ctx.paths.dataset_registry_path())
    except FileNotFoundError:
        logger.warning("Dataset registry not found")
    
    dataset_info = registry.get(cfg.data.dataset_name,None)

    # create loaders
    data_cfg = cfg.data.resolved(
        data_dir=data_dir,
        norm_prm=norm_prm,
        dataset_info=dataset_info,
        volumetric=ctx.volumetric,
    )

    train_loader, val_loader, model_norm_prm, patch_info = make_training_loaders(
        config=data_cfg,
    )

    loaders = SimpleNamespace(train=train_loader, val=val_loader)
    meta = SimpleNamespace(model_norm_prm=model_norm_prm, patch_info=patch_info)

    return loaders, meta
