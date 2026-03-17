"""Training run-directory and config persistence helpers.

This module owns the parts of training setup that interact with the filesystem:
creating or reusing run directories, copying retrain origin artifacts, and
saving the effective training config once runtime-derived normalization is
known.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from lisai.config import prune_config_for_saving, save_yaml
from lisai.infra.fs import create_run_dir

if TYPE_CHECKING:
    from lisai.config.models import ResolvedExperiment
    from lisai.training.runtime import TrainingRuntime



def _safe_copy(src: Path, dst: Path):
    """Copy a file when present, ignoring missing files and copy failures."""
    try:
        if src.exists():
            shutil.copyfile(src, dst)
    except Exception:
        pass



def prepare_run_dir(
    cfg: ResolvedExperiment,
    runtime: TrainingRuntime,
) -> tuple[Path | None, str]:
    """Resolve the effective training run directory and runtime run name.

    Semantics:
    - if saving is disabled, no run directory is created
    - if mode is `continue_training`, the origin run directory is reused
    - otherwise a new canonical run directory is created
    - if mode is `retrain`, origin artifacts are copied into `retrain_origin/`
    """
    if not bool(cfg.saving.enabled):
        return None, cfg.experiment.exp_name

    mode = cfg.experiment.mode

    origin_dir = None
    if mode in {"continue_training", "retrain"}:
        if not cfg.experiment.origin_run_dir:
            raise ValueError(f"Mode '{mode}' requires experiment.origin_run_dir")
        origin_dir = Path(cfg.experiment.origin_run_dir)

    if mode == "continue_training":
        return origin_dir, origin_dir.name

    run_dir, exp_name = create_run_dir(
        paths=runtime.paths,
        ds_name=cfg.data.dataset_name,
        exp_name=cfg.experiment.exp_name,
        subfolder=cfg.routing.models_subfolder,
        overwrite=bool(cfg.experiment.overwrite),
    )

    runtime.paths.checkpoints_dir(run_dir=run_dir).mkdir(parents=True, exist_ok=True)
    runtime.paths.validation_images_dir(run_dir=run_dir).mkdir(parents=True, exist_ok=True)

    if mode == "retrain":
        retrain_origin_dir = runtime.paths.retrain_origin_dir(run_dir=run_dir)
        retrain_origin_dir.mkdir(parents=True, exist_ok=False)

        _safe_copy(
            runtime.paths.loss_file_path(run_dir=origin_dir),
            runtime.paths.retrain_origin_loss_path(run_dir=run_dir),
        )
        _safe_copy(
            runtime.paths.log_file_path(run_dir=origin_dir),
            runtime.paths.retrain_origin_log_path(run_dir=run_dir),
        )
        _safe_copy(
            runtime.paths.cfg_train_path(run_dir=origin_dir),
            runtime.paths.retrain_origin_cfg_path(run_dir=run_dir),
        )

    return run_dir, exp_name



def save_training_config(
    cfg: ResolvedExperiment,
    runtime: TrainingRuntime,
    data_norm_prm: dict | None = None,
    model_norm_prm: dict | None = None,
) -> Path | None:
    """Persist the effective saved training config for a new or retrain run."""
    if runtime.run_dir is None or cfg.experiment.mode == "continue_training":
        return None

    clean_cfg = prune_config_for_saving(cfg)

    if data_norm_prm is not None:
        normalization = dict(clean_cfg.get("normalization") or {})
        normalization["norm_prm"] = dict(data_norm_prm)
        clean_cfg["normalization"] = normalization

    if model_norm_prm is not None:
        clean_cfg["model_norm_prm"] = dict(model_norm_prm)

    cfg_path = runtime.paths.cfg_train_path(run_dir=runtime.run_dir)
    save_yaml(clean_cfg, cfg_path)
    return cfg_path
