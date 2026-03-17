"""Config-side evaluation helpers for loading a saved training run.

This module owns the evaluation boundary that turns a saved `config_train.yaml`
into a small, immutable `SavedTrainingRun` object. It is intentionally separate
from the live inference runtime so evaluation keeps a clean split between
saved configuration and process-time resources.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

from lisai.config import load_yaml, settings
from lisai.config.models import ResolvedExperiment
from lisai.defaults import DEFAULT_TILING_SIZE
from lisai.infra.paths import Paths

CheckpointMethod = Literal["state_dict", "full_model"]



def resolve_run_dir(*, dataset_name: str, subfolder: str, exp_name: str) -> Path:
    """Resolve the canonical training run directory from experiment routing fields."""
    paths = Paths(settings)
    return paths.run_dir(dataset_name=dataset_name, models_subfolder=subfolder, exp_name=exp_name)



def _noise_model_name(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        name = value.get("name")
        return str(name) if name is not None else None
    name = getattr(value, "name", None)
    return str(name) if name is not None else None



def _checkpoint_methods(cfg: ResolvedExperiment) -> tuple[CheckpointMethod, ...]:
    """Extract the checkpoint formats that the saved run can be loaded from."""
    methods: list[CheckpointMethod] = []
    if bool(cfg.saving.state_dict):
        methods.append("state_dict")
    if bool(cfg.saving.entire_model):
        methods.append("full_model")
    if not methods:
        raise ValueError("Saved training config does not enable any checkpoint format.")
    return tuple(methods)



def _default_tiling_size(architecture: str) -> int | None:
    """Resolve the default inference tiling size for a model architecture."""
    for key in (architecture, architecture.replace("_", "")):
        if key in DEFAULT_TILING_SIZE:
            return int(DEFAULT_TILING_SIZE[key])
    return None



@dataclass(frozen=True)
class SavedTrainingRun:
    """Immutable evaluation-facing summary of one saved training run.

    The object keeps only the subset of the saved training configuration that
    evaluation needs: model description, normalization settings, checkpoint
    formats, and data-loading hints.
    """

    run_dir: Path
    experiment_name: str
    dataset_name: str
    data_subfolder: str
    data_cfg: dict[str, Any]
    model_architecture: str
    model_parameters: dict[str, Any]
    data_norm_prm: dict[str, Any] | None
    model_norm_prm: dict[str, Any] | None
    noise_model_name: str | None
    checkpoint_methods: tuple[CheckpointMethod, ...]
    patch_size: int | None
    downsamp_factor: int
    upsampling_factor: int
    context_length: int | None
    default_tiling_size: int | None

    @property
    def is_lvae(self) -> bool:
        """Whether this saved run uses the LVAE architecture."""
        return self.model_architecture == "lvae"

    @classmethod
    def from_resolved(cls, cfg: ResolvedExperiment, *, run_dir: Path) -> "SavedTrainingRun":
        """Project a validated `ResolvedExperiment` into the evaluation config boundary."""
        architecture = cfg.model.architecture
        if not architecture:
            raise ValueError("Saved training config is missing model.architecture.")

        model_parameters = dict(cfg.model.parameters or {})
        norm_prm = cfg.normalization.get("norm_prm")
        data_norm_prm = dict(norm_prm) if isinstance(norm_prm, Mapping) else None
        model_norm_prm = dict(cfg.model_norm_prm) if isinstance(cfg.model_norm_prm, Mapping) else None
        upsamp = model_parameters.get("upsamp")
        if upsamp is None:
            upsamp = model_parameters.get("upsampling_factor")
        context_length = None
        if cfg.data.timelapse_prm is not None:
            context_length = cfg.data.timelapse_prm.context_length

        return cls(
            run_dir=Path(run_dir),
            experiment_name=Path(run_dir).name,
            dataset_name=cfg.data.dataset_name,
            data_subfolder=cfg.routing.data_subfolder,
            data_cfg=cfg.data.model_dump(exclude_none=True),
            model_architecture=architecture,
            model_parameters=model_parameters,
            data_norm_prm=data_norm_prm,
            model_norm_prm=model_norm_prm,
            noise_model_name=_noise_model_name(cfg.noise_model),
            checkpoint_methods=_checkpoint_methods(cfg),
            patch_size=cfg.data.model_patch_size,
            downsamp_factor=cfg.data.downsampling_factor,
            upsampling_factor=int(upsamp) if upsamp is not None else 1,
            context_length=int(context_length) if context_length is not None else None,
            default_tiling_size=_default_tiling_size(architecture),
        )



def load_saved_run(run_dir: Path) -> SavedTrainingRun:
    """Load, validate, and project a saved training config from a run directory."""
    run_dir = Path(run_dir)
    cfg_path = Paths(settings).cfg_train_path(run_dir=run_dir)
    cfg = ResolvedExperiment.model_validate(load_yaml(cfg_path))
    return SavedTrainingRun.from_resolved(cfg, run_dir=run_dir)



__all__ = ["CheckpointMethod", "SavedTrainingRun", "load_saved_run", "resolve_run_dir"]
