from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from lisai.infra.config import prune_config_for_saving, save_yaml
from lisai.infra.fs import create_run_dir

if TYPE_CHECKING:
    from lisai.infra.config.schema import ResolvedExperiment

    from .context import TrainingContext

def _safe_copy(src: Path, dst: Path):
    try:
        if src.exists():
            shutil.copyfile(src, dst)
    except Exception:
        pass


def prepare_run_dir(
    cfg: ResolvedExperiment,
    ctx: TrainingContext,
) -> tuple[Path | None, str]:
    """
    Run directory semantics:
      - if saving disabled: ctx.run_dir=None
      - if continue_training: reuse origin run directory
      - else create a new directory under canonical model root
      - if retrain: copy origin artifacts into retrain_origin/
      - save cleaned config into the run dir (config_train filename)
    """
    # saving disabled => no run dir
    if not bool(cfg.saving.enabled):
        return None, cfg.experiment.exp_name

    mode = cfg.experiment.mode

    # origin folder is computed once in resolve_config
    origin_dir = None
    if mode in {"continue_training", "retrain"}:
        if not cfg.experiment.origin_run_dir:
            raise ValueError(f"Mode '{mode}' requires experiment.origin_run_dir")
        origin_dir = Path(cfg.experiment.origin_run_dir)

    # continue_training: reuse origin folder
    if mode == "continue_training":
        return origin_dir, origin_dir.name

    # train / retrain: create new folder
    run_dir, exp_name = create_run_dir(
        paths=ctx.paths,
        ds_name=cfg.data.dataset_name,
        exp_name=cfg.experiment.exp_name,
        subfolder=cfg.routing.models_subfolder,
        overwrite=bool(cfg.experiment.overwrite),
    )

    # Create artifact subfolders (config-driven via settings.project.run_layout)
    ctx.paths.checkpoints_dir(run_dir=run_dir).mkdir(parents=True, exist_ok=True)
    ctx.paths.validation_images_dir(run_dir=run_dir).mkdir(parents=True, exist_ok=True)

    # retrain: copy origin artifacts
    if mode == "retrain":
        retrain_origin_dir = ctx.paths.retrain_origin_dir(run_dir=run_dir)
        retrain_origin_dir.mkdir(parents=True, exist_ok=False)

        _safe_copy(
            ctx.paths.loss_file_path(run_dir=origin_dir),
            ctx.paths.retrain_origin_loss_path(run_dir=run_dir),
        )
        _safe_copy(
            ctx.paths.log_file_path(run_dir=origin_dir),
            ctx.paths.retrain_origin_log_path(run_dir=run_dir),
        )
        _safe_copy(
            ctx.paths.cfg_train_path(run_dir=origin_dir),
            ctx.paths.retrain_origin_cfg_path(run_dir=run_dir),
        )

    # Save cleaned config
    clean_cfg = prune_config_for_saving(cfg)
    cfg_path = ctx.paths.cfg_train_path(run_dir=run_dir)
    save_yaml(clean_cfg, cfg_path)

    return run_dir, exp_name
