from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, Literal, Mapping, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

Mode = Literal["train", "continue_training", "retrain"]
CheckpointMethod = Literal["state_dict", "full_model"]


class ExperimentSection(BaseModel):
    model_config = ConfigDict(extra="allow")
    mode: Mode = "train"
    exp_name: str = "unnamed_experiment"
    overwrite: bool = False
    origin_run_dir: Optional[str] = None


class RoutingSection(BaseModel):
    model_config = ConfigDict(extra="allow")
    data_subfolder: str = ""
    models_subfolder: str = ""
    tensorboard_subfolder: Optional[str] = None
    inference_subfolder: str = ""

    @model_validator(mode="after")
    def _defaults(self):
        if self.tensorboard_subfolder is None:
            self.tensorboard_subfolder = self.models_subfolder
        return self


class DataSection(BaseModel):
    model_config = ConfigDict(extra="allow")

    dataset_name: str = "unknown_dataset"
    canonical_load: bool = True

    # Core loader setup
    prep_before: bool = True
    already_split: bool = True
    paired: bool = False
    input: Optional[str] = None
    target: Optional[str] = None

    # Legacy aliases still seen in older configs
    inp: Optional[str] = None
    gt: Optional[str] = None

    # Data format selection and filtering
    data_format: Optional[str] = None
    filters: list[str] = Field(default_factory=lambda: ["tif", "tiff"])

    # Patching and batching
    batch_size: int = 1
    patch_size: Optional[int] = None
    val_patch_size: Optional[int] = None
    patch_thresh: Optional[float] = None
    select_on_gt: bool = False
    augmentation: bool = False
    initial_crop: Optional[Any] = None
    mltpl_noise: bool = False

    # Transform parameters
    masking: Optional[Dict[str, Any]] = None
    downsampling: Optional[Dict[str, Any]] = None
    artificial_movement: Optional[Dict[str, Any]] = None
    inp_transform: Optional[Dict[str, Any]] = None
    gt_transform: Optional[Dict[str, Any]] = None
    timelapse_prm: Optional[Dict[str, Any]] = None
    mltpl_snr_prm: Optional[Dict[str, Any]] = None

    # Normalization inputs
    norm_prm: Optional[Dict[str, Any]] = None
    model_norm_prm: Optional[Dict[str, Any]] = None
    avgObs_per_noise: Optional[list[float]] = None
    stdObs_per_noise: Optional[list[float]] = None

    # Runtime flag propagated from system setup
    volumetric: bool = False
    data_dir: Optional[Path] = Field(default=None, exclude=True)
    dataset_info: Optional[Dict[str, Any]] = Field(default=None, exclude=True)

    @model_validator(mode="after")
    def _normalize_aliases(self):
        if self.target is None and self.gt is not None:
            self.target = self.gt
        if self.input is None and self.inp is not None:
            self.input = self.inp
        if self.paired and not self.target:
            raise ValueError("Paired dataset requires `target`.")
        if self.batch_size <= 0:
            raise ValueError("`batch_size` must be > 0.")
        if self.patch_size is not None and self.patch_size <= 0:
            raise ValueError("`patch_size` must be > 0.")
        if self.val_patch_size is not None and self.val_patch_size <= 0:
            raise ValueError("`val_patch_size` must be > 0.")
        return self

    @property
    def resolved_data_format(self) -> str:
        if self.dataset_info is not None and self.dataset_info.get("data_format") is not None:
            return self.dataset_info["data_format"]
        if self.data_format is not None:
            return self.data_format
        warnings.warn("Data format not specified, put to 'single' by default.")
        return "single"

    def resolved(
        self,
        *,
        data_dir: Path,
        dataset_info: Optional[Mapping[str, Any]] = None,
        norm_prm: Optional[Mapping[str, Any]] = None,
        model_norm_prm: Optional[Mapping[str, Any]] = None,
        split: Optional[str] = None,
        volumetric: Optional[bool] = None,
    ) -> "DataSection":
        """
        Return a resolved copy for data loading, with runtime fields injected.
        # """
        # if not self.input:
        #     raise ValueError("`data.input` must be provided for data loading.")

        updates: Dict[str, Any] = {"data_dir": Path(data_dir)}
        if dataset_info is not None:
            updates["dataset_info"] = dict(dataset_info)
        if norm_prm is not None:
            updates["norm_prm"] = dict(norm_prm)
        if model_norm_prm is not None:
            updates["model_norm_prm"] = dict(model_norm_prm)
        if split is not None:
            updates["split"] = split
        if volumetric is not None:
            updates["volumetric"] = volumetric

        return self.model_copy(update=updates, deep=True)


class ModelSection(BaseModel):
    model_config = ConfigDict(extra="allow")
    architecture: str = ""
    parameters: Dict[str, Any] = Field(default_factory=dict)


class TrainingSection(BaseModel):
    model_config = ConfigDict(extra="allow")
    n_epochs: int = 1
    batch_size: int = 1
    learning_rate: float = 1e-4
    optimizer: str = "Adam"
    scheduler: Optional[str] = None
    progress_bar: bool = False
    early_stop: bool = False
    pos_encod: bool = False


class SavingSection(BaseModel):
    model_config = ConfigDict(extra="allow")
    enabled: bool = True
    canonical_save: bool = True
    validation_images: bool = True
    validation_freq: int = 10


class TensorboardSection(BaseModel):
    model_config = ConfigDict(extra="allow")
    enabled: bool = False


class LoadCheckpoint(BaseModel):
    model_config = ConfigDict(extra="allow")
    method: Optional[CheckpointMethod] = None
    selector: Optional[str] = None  # "best" | "last" | "epoch" | etc (keep flexible)
    epoch: Optional[int] = None
    filename: Optional[str] = None


class LoadModelSection(BaseModel):
    model_config = ConfigDict(extra="allow")
    enabled: bool = False
    source: Optional[str] = None     # "canonical" | "path"
    run_dir: Optional[str] = None    # resolved absolute
    checkpoint: LoadCheckpoint = Field(default_factory=LoadCheckpoint)


class ResolvedExperiment(BaseModel):
    """
    This is what resolve_config returns.
    It is intentionally permissive (extra=allow) on sections that evolve.
    """
    model_config = ConfigDict(extra="allow")

    experiment: ExperimentSection = Field(default_factory=ExperimentSection)
    routing: RoutingSection = Field(default_factory=RoutingSection)
    data: DataSection = Field(default_factory=DataSection)
    model: ModelSection = Field(default_factory=ModelSection)
    training: TrainingSection = Field(default_factory=TrainingSection)
    normalization: Dict[str, Any] = Field(default_factory=dict)
    loss_function: Dict[str, Any] = Field(default_factory=dict)

    saving: SavingSection = Field(default_factory=SavingSection)
    tensorboard: TensorboardSection = Field(default_factory=TensorboardSection)
    load_model: LoadModelSection = Field(default_factory=LoadModelSection)
