from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

Mode = Literal["train", "continue_training", "retrain"]


class ExperimentSection(BaseModel):
    """High-level experiment metadata controlling how training is launched."""

    model_config = ConfigDict(extra="allow")

    mode: Mode = Field(
        default="train",
        description="Training lifecycle mode: start a new run, continue an existing one, or retrain from loaded weights.",
    )
    exp_name: str = Field(
        default="unnamed_experiment",
        description="Experiment name used to create the output run directory and derived artifact names.",
    )
    overwrite: bool = Field(
        default=False,
        description="Whether an existing experiment directory with the same name may be overwritten.",
    )
    post_training_inference: bool = Field(
        default=True,
        description="Whether to trigger automatic post-training evaluation when training completes or stops early.",
    )


class ResolvedExperimentSection(ExperimentSection):
    """Resolved experiment metadata enriched with runtime-only bookkeeping."""

    origin_run_dir: Optional[str] = Field(
        default=None,
        description="Original run directory used as the source when continuing training or retraining.",
    )


class RoutingSection(BaseModel):
    """Subfolder routing used to place datasets, checkpoints, logs, and inference outputs."""

    model_config = ConfigDict(extra="allow")

    data_subfolder: str = Field(
        default="",
        description="Dataset subfolder selected under the configured data root.",
    )
    models_subfolder: str = Field(
        default="",
        description="Subfolder under the models root where checkpoints and training artifacts are stored.",
    )
    tensorboard_subfolder: Optional[str] = Field(
        default=None,
        description="Subfolder under the tensorboard root. Defaults to the models_subfolder when omitted.",
    )
    inference_subfolder: str = Field(
        default="",
        description="Subfolder under the inference root where evaluation or apply outputs are written.",
    )

    @model_validator(mode="after")
    def _defaults(self):
        if self.tensorboard_subfolder is None:
            self.tensorboard_subfolder = self.models_subfolder
        return self


class TrainingSection(BaseModel):
    """Core optimization settings for the training loop."""

    model_config = ConfigDict(extra="allow")

    n_epochs: int = Field(
        default=1,
        description="Maximum number of training epochs to run.",
    )
    batch_size: int = Field(
        default=1,
        description="Fallback batch size used by the trainer when the data section does not provide one.",
    )
    learning_rate: float = Field(
        default=1e-4,
        description="Initial learning rate passed to the optimizer.",
    )
    optimizer: str = Field(
        default="Adam",
        description="Optimizer name used to update the model parameters.",
    )
    scheduler: str | None = Field(
        default=None,
        description="Optional learning-rate scheduler name. Use null to disable scheduling.",
    )
    progress_bar: bool = Field(
        default=False,
        description="Whether to show a live progress bar during training iterations.",
    )
    early_stop: bool = Field(
        default=False,
        description="Whether early stopping is enabled when validation performance stops improving.",
    )
    pos_encod: bool = Field(
        default=False,
        description="Whether positional encoding should be enabled for models that support it.",
    )


class SavingSection(BaseModel):
    """Checkpoint and validation-image saving behavior."""

    model_config = ConfigDict(extra="allow")

    enabled: bool = Field(
        default=True,
        description="Whether checkpoints and other training artifacts should be written to disk.",
    )
    canonical_save: bool = Field(
        default=True,
        description="Whether checkpoint/output paths should be resolved through the canonical project routing rules.",
    )
    validation_images: bool = Field(
        default=True,
        description="Whether to save validation image previews during training.",
    )
    validation_freq: int = Field(
        default=10,
        description="Number of epochs between validation-image saves.",
    )
    state_dict: bool = Field(
        default=False,
        description="Whether checkpoints should be saved as state_dict files.",
    )
    entire_model: bool = Field(
        default=False,
        description="Whether checkpoints should be saved as serialized full-model files.",
    )
    overwrite_best: bool = Field(
        default=True,
        description="Whether the current best checkpoint may overwrite the previously saved best checkpoint.",
    )


class TensorboardSection(BaseModel):
    """TensorBoard logging settings."""

    model_config = ConfigDict(extra="allow")

    enabled: bool = Field(
        default=False,
        description="Whether TensorBoard summaries should be written during training.",
    )


class NoiseModelSection(BaseModel):
    """Optional auxiliary noise-model configuration."""

    model_config = ConfigDict(extra="allow")

    name: str | None = Field(
        default=None,
        description="Registered noise-model name to attach to the experiment. Use null to disable it.",
    )
