from __future__ import annotations

from typing import Any, Dict

from pydantic import BaseModel, ConfigDict, Field

from .data import DataSection, ExperimentDataSection
from .loading import ExperimentLoadModelSection, LoadModelSection
from .loss import LossFunctionConfig
from .model import ModelSection
from .normalization import NormalizationSection
from .sections import (
    ExperimentSection,
    NoiseModelSection,
    ResolvedExperimentSection,
    RoutingSection,
    SavingSection,
    TensorboardSection,
    TrainingSection,
)


class ExperimentConfig(BaseModel):
    """User-authored training experiment YAML schema."""

    model_config = ConfigDict(extra="allow")

    experiment: ExperimentSection = Field(
        default_factory=ExperimentSection,
        description="High-level experiment metadata and execution mode.",
    )
    routing: RoutingSection = Field(
        default_factory=RoutingSection,
        description="Subfolder routing that determines where data, models, logs, and inference outputs live.",
    )
    data: ExperimentDataSection = Field(
        default_factory=ExperimentDataSection,
        description="User-authored data-loading, patching, and preprocessing settings.",
    )
    model: ModelSection | None = Field(
        default=None,
        description="Model architecture selection and typed constructor parameters.",
    )
    training: TrainingSection = Field(
        default_factory=TrainingSection,
        description="Core optimization settings for the training loop.",
    )
    normalization: NormalizationSection = Field(
        default_factory=NormalizationSection,
        description="Dataset normalization settings applied before or during data loading.",
    )
    model_norm_prm: Dict[str, Any] | None = Field(
        default=None,
        description="Optional model-output normalization settings kept separate from the main data normalization block.",
    )
    loss_function: LossFunctionConfig | None = Field(
        default=None,
        description="Typed loss-function configuration used to instantiate the trainer loss.",
    )
    noise_model: NoiseModelSection | str | None = Field(
        default=None,
        description="Optional noise-model configuration or shortcut name.",
    )
    saving: SavingSection = Field(
        default_factory=SavingSection,
        description="Checkpoint and validation-image saving behavior.",
    )
    tensorboard: TensorboardSection = Field(
        default_factory=TensorboardSection,
        description="TensorBoard logging settings.",
    )
    load_model: ExperimentLoadModelSection = Field(
        default_factory=ExperimentLoadModelSection,
        description="Optional source-run selection used to initialize the model from a previous checkpoint.",
    )


class ResolvedExperiment(BaseModel):
    """Runtime training config after merge and normalization."""

    model_config = ConfigDict(extra="allow")

    experiment: ResolvedExperimentSection = Field(
        default_factory=ResolvedExperimentSection,
        description="Resolved experiment metadata enriched with runtime-only bookkeeping.",
    )
    routing: RoutingSection = Field(
        default_factory=RoutingSection,
        description="Resolved routing configuration for datasets, models, logs, and inference outputs.",
    )
    data: DataSection = Field(
        default_factory=DataSection,
        description="Resolved data-loading and preprocessing settings used by the runtime.",
    )
    model: ModelSection | None = Field(
        default=None,
        description="Resolved model architecture selection and typed constructor parameters.",
    )
    training: TrainingSection = Field(
        default_factory=TrainingSection,
        description="Resolved optimization settings for the training loop.",
    )
    normalization: NormalizationSection = Field(
        default_factory=NormalizationSection,
        description="Resolved dataset normalization settings used during training data preparation.",
    )
    model_norm_prm: Dict[str, Any] | None = Field(
        default=None,
        description="Resolved model-output normalization settings.",
    )
    loss_function: LossFunctionConfig | None = Field(
        default=None,
        description="Resolved typed loss-function configuration.",
    )
    noise_model: NoiseModelSection | str | None = Field(
        default=None,
        description="Resolved optional noise-model configuration or shortcut name.",
    )
    saving: SavingSection = Field(
        default_factory=SavingSection,
        description="Resolved checkpoint and artifact-saving behavior.",
    )
    tensorboard: TensorboardSection = Field(
        default_factory=TensorboardSection,
        description="Resolved TensorBoard logging settings.",
    )
    load_model: LoadModelSection = Field(
        default_factory=LoadModelSection,
        description="Resolved source-run and checkpoint selection used when loading a prior model.",
    )
