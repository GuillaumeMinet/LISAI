from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from lisai.config.models import ResolvedExperiment


@dataclass(frozen=True)
class ModelSpec:
    # build
    architecture: str
    parameters: dict[str, Any]

    # mode
    mode: str  # "train" | "continue_training" | "retrain"

    # LVAE / normalization
    normalization: dict[str, Any]
    noise_model_name: Optional[str] = None

    # data-derived (needed for LVAE img_shape)
    patch_size: Optional[int] = None
    downsamp_factor: int = 1

    # load
    origin_run_dir: Optional[Path] = None
    checkpoint_method: Optional[str] = None   # "state_dict" | "full_model" | None
    checkpoint_selector: Optional[str] = None # "best" | "last" | "epoch" | None
    checkpoint_epoch: Optional[int] = None
    checkpoint_filename: Optional[str] = None


@dataclass(frozen=True)
class InferenceSpec:
    run_dir: Path

    # build
    architecture: str
    parameters: dict[str, Any]

    # LVAE / normalization
    normalization: dict[str, Any]
    model_norm_prm: dict[str, Any] | None = None
    noise_model_name: Optional[str] = None

    # data-derived (needed for LVAE img_shape)
    patch_size: Optional[int] = None
    downsamp_factor: int = 1

    # load
    checkpoint_method: str = "state_dict"
    checkpoint_selector: str = "best"
    checkpoint_epoch: Optional[int] = None


@dataclass(frozen=True)
class RunSpec:
    cfg: ResolvedExperiment

    @property
    def mode(self) -> str:
        return self.cfg.experiment.mode

    @property
    def exp_name(self) -> str:
        return self.cfg.experiment.exp_name

    @property
    def dataset_name(self) -> str:
        return self.cfg.data.dataset_name

    @property
    def should_load(self) -> bool:
        return self.mode in {"continue_training", "retrain"}

    @property
    def origin_run_dir(self) -> Optional[Path]:
        v = self.cfg.experiment.origin_run_dir
        return Path(v) if v else None

    @property
    def model_architecture(self) -> str:
        return self.cfg.model.architecture

    @property
    def noise_model_name(self) -> Optional[str]:
        noise_model = getattr(self.cfg, "noise_model", None)
        if isinstance(noise_model, dict):
            return noise_model.get("name")
        if isinstance(noise_model, str):
            return noise_model
        return getattr(noise_model, "name", None)

    def model_spec(self) -> ModelSpec:
        patch_size = self.cfg.data.model_patch_size
        ds_factor = self.cfg.data.downsampling_factor

        ckpt = self.cfg.load_model.checkpoint if self.cfg.load_model else None

        return ModelSpec(
            architecture=self.cfg.model.architecture,
            parameters=self.cfg.model.parameters or {},
            mode=self.cfg.experiment.mode,
            normalization=self.cfg.normalization or {},
            noise_model_name=self.noise_model_name,
            patch_size=int(patch_size) if patch_size is not None else None,
            downsamp_factor=int(ds_factor) if ds_factor is not None else 1,
            origin_run_dir=self.origin_run_dir,
            checkpoint_method=getattr(ckpt, "method", None) if ckpt else None,
            checkpoint_selector=getattr(ckpt, "selector", None) if ckpt else None,
            checkpoint_epoch=getattr(ckpt, "epoch", None) if ckpt else None,
            checkpoint_filename=getattr(ckpt, "filename", None) if ckpt else None,
        )
