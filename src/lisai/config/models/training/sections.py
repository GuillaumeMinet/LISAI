from __future__ import annotations

from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

Mode = Literal["train", "continue_training", "retrain"]


class ExperimentSection(BaseModel):
    model_config = ConfigDict(extra="allow")
    mode: Mode = "train"
    exp_name: str = "unnamed_experiment"
    overwrite: bool = False
    post_training_inference: bool = True


class ResolvedExperimentSection(ExperimentSection):
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
    scheduler: str | None = None
    progress_bar: bool = False
    early_stop: bool = False
    pos_encod: bool = False


class SavingSection(BaseModel):
    model_config = ConfigDict(extra="allow")
    enabled: bool = True
    canonical_save: bool = True
    validation_images: bool = True
    validation_freq: int = 10
    state_dict: bool = False
    entire_model: bool = False
    overwrite_best: bool = True


class TensorboardSection(BaseModel):
    model_config = ConfigDict(extra="allow")
    enabled: bool = False


class NoiseModelSection(BaseModel):
    model_config = ConfigDict(extra="allow")
    name: str | None = None

