from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lisai.models.loader import prepare_model_for_training

from .noise_model import resolve_noise_model_name

if TYPE_CHECKING:
    from lisai.config.models import ResolvedExperiment


@dataclass(frozen=True)
class TrainingModelSpec:
    """Training-local contract for model build/load decisions."""

    architecture: str
    parameters: dict[str, Any]
    mode: str
    patch_size: int | None = None
    downsamp_factor: int = 1
    origin_run_dir: Path | None = None
    checkpoint_method: str | None = None
    checkpoint_selector: str | None = None
    checkpoint_epoch: int | None = None
    checkpoint_filename: str | None = None
    noise_model_name: str | None = None

    @classmethod
    def from_config(cls, cfg: ResolvedExperiment) -> TrainingModelSpec:
        ckpt = cfg.load_model.checkpoint if cfg.load_model else None
        origin_run_dir = cfg.experiment.origin_run_dir
        patch_size = cfg.data.model_patch_size
        downsamp_factor = cfg.data.downsampling_factor

        return cls(
            architecture=cfg.model.architecture,
            parameters=cfg.model.parameters or {},
            mode=cfg.experiment.mode,
            patch_size=int(patch_size) if patch_size is not None else None,
            downsamp_factor=int(downsamp_factor) if downsamp_factor is not None else 1,
            origin_run_dir=Path(origin_run_dir) if origin_run_dir else None,
            checkpoint_method=getattr(ckpt, "method", None) if ckpt else None,
            checkpoint_selector=getattr(ckpt, "selector", None) if ckpt else None,
            checkpoint_epoch=getattr(ckpt, "epoch", None) if ckpt else None,
            checkpoint_filename=getattr(ckpt, "filename", None) if ckpt else None,
            noise_model_name=resolve_noise_model_name(cfg),
        )



def build_model(cfg: ResolvedExperiment, device, model_norm_prm, noise_model):
    """Build and optionally load the training model from the resolved config."""
    logger = logging.getLogger("lisai")
    spec = TrainingModelSpec.from_config(cfg)

    model, state = prepare_model_for_training(
        spec=spec,
        device=device,
        model_norm_prm=model_norm_prm,
        noise_model=noise_model,
    )

    logger.info(f"Model initialized: {type(model).__name__}")
    return model, state
