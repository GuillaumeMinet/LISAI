from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from lisai.infra.config import settings
from lisai.infra.fs import create_tb_folder
from lisai.infra.logging import setup_logger
from lisai.infra.paths import Paths
from lisai.runtime.spec import RunSpec
from lisai.training.callbacks import TensorBoardCallback, ValidationImagesCallback

from .context import TrainingContext
from .run_dir import prepare_run_dir

if TYPE_CHECKING:
    from lisai.infra.config.schema import ResolvedExperiment

def initialize(cfg: ResolvedExperiment) -> TrainingContext:
    """
    Sets up the environment: paths, logging, device, tensorboard.
    cfg is a ResolvedExperiment (Pydantic).
    Return ctx: a typed context object holding all infrastructure state.
    """
    paths = Paths(settings)
    spec = RunSpec(cfg)

    # init context
    ctx = TrainingContext(
        cfg=cfg,
        spec=spec,
        paths=paths,
        exp_name=cfg.experiment.exp_name,
        mode=cfg.experiment.mode,
        volumetric=(cfg.model.architecture == "unet3d"),
    )

    # saving / run dir
    run_dir, exp_name = prepare_run_dir(cfg, ctx)
    ctx.run_dir = run_dir
    ctx.exp_name = exp_name  # prepare_run_dir may have changed it to have a unique exp_name

    # Setup logger: console + log file
    log_file = None
    if run_dir is not None:
        log_file = paths.log_file_path(run_dir=run_dir)

    use_tqdm = bool(getattr(cfg.training, "progress_bar", False))
    logger, console_filter, file_filter = setup_logger(
        name="lisai",
        level=logging.INFO,
        log_file=log_file,
        use_tqdm=use_tqdm,
        file_format="%(asctime)s %(message)s",
        file_datefmt="%Y-%m-%d %H:%M:%S",
        file_enabled=False,
    )

    ctx.logger = logger
    ctx.console_filter = console_filter
    ctx.file_filter = file_filter

    def _enable_console(enable: bool):
        console_filter.enable = bool(enable)

    def _enable_file(enable: bool):
        file_filter.enable = bool(enable)

    ctx.enable_console_logs = _enable_console
    ctx.enable_file_logs = _enable_file

    logger.info(f"Training Initialization: {exp_name}")
    if run_dir is not None:
        logger.info(f"Saving to: {run_dir}")
    else:
        logger.warning("Saving disabled!")

    # Setup tensorboard
    writer = None
    if bool(cfg.tensorboard.enabled):
        try:
            from torch.utils.tensorboard import SummaryWriter
        except Exception as e:
            logger.warning(f"TensorBoard requested but unavailable: {type(e).__name__}: {e}")
            SummaryWriter = None

        if SummaryWriter is not None:
            tb_root = paths.tensorboard_dir(
                dataset_name=cfg.data.dataset_name,
                tensorboard_subfolder=cfg.routing.tensorboard_subfolder or cfg.routing.models_subfolder,
            )
            tb_folder, _ = create_tb_folder(tb_root, exp_name, exist_ok=(run_dir is not None))
            writer = SummaryWriter(log_dir=str(tb_folder))

    ctx.writer = writer

    # Setup device
    device_name = getattr(cfg.training, "device", "cuda")
    if device_name == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available. Switching to CPU.")
        device_name = "cpu"
    ctx.device = torch.device(device_name)

    # Callbacks
    callbacks = []

    if writer is not None:
        callbacks.append(TensorBoardCallback(writer, volumetric=ctx.volumetric))

    images_dir = None
    if ctx.run_dir is not None:
        images_dir = ctx.paths.validation_images_dir(run_dir=ctx.run_dir)

    if images_dir is not None:
        callbacks.append(
            ValidationImagesCallback(
                images_dir=images_dir,
                enabled=bool(getattr(cfg.saving, "validation_images", True)),
                freq=int(getattr(cfg.saving, "validation_freq", 10)),
                volumetric=ctx.volumetric,
            )
        )

    ctx.callbacks = callbacks
    return ctx
