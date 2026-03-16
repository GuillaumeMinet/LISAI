from __future__ import annotations

from typing import Any, Dict

from pydantic import BaseModel, ConfigDict, Field

from .data import DataSection, ExperimentDataSection
from .loading import ExperimentLoadModelSection, LoadModelSection
from .sections import (
    ExperimentSection,
    ModelSection,
    NoiseModelSection,
    ResolvedExperimentSection,
    RoutingSection,
    SavingSection,
    TensorboardSection,
    TrainingSection,
)


class ExperimentConfig(BaseModel):
    """
    User-authored training experiment YAML schema.
    """

    model_config = ConfigDict(extra="allow")

    experiment: ExperimentSection = Field(default_factory=ExperimentSection)
    routing: RoutingSection = Field(default_factory=RoutingSection)
    data: ExperimentDataSection = Field(default_factory=ExperimentDataSection)
    model: ModelSection = Field(default_factory=ModelSection)
    training: TrainingSection = Field(default_factory=TrainingSection)
    normalization: Dict[str, Any] = Field(default_factory=dict)
    loss_function: Dict[str, Any] = Field(default_factory=dict)
    noise_model: NoiseModelSection | str | None = None
    saving: SavingSection = Field(default_factory=SavingSection)
    tensorboard: TensorboardSection = Field(default_factory=TensorboardSection)
    load_model: ExperimentLoadModelSection = Field(default_factory=ExperimentLoadModelSection)


class ResolvedExperiment(BaseModel):
    """
    Runtime training config after merge + normalization.
    """

    model_config = ConfigDict(extra="allow")

    experiment: ResolvedExperimentSection = Field(default_factory=ResolvedExperimentSection)
    routing: RoutingSection = Field(default_factory=RoutingSection)
    data: DataSection = Field(default_factory=DataSection)
    model: ModelSection = Field(default_factory=ModelSection)
    training: TrainingSection = Field(default_factory=TrainingSection)
    normalization: Dict[str, Any] = Field(default_factory=dict)
    loss_function: Dict[str, Any] = Field(default_factory=dict)
    noise_model: NoiseModelSection | str | None = None
    saving: SavingSection = Field(default_factory=SavingSection)
    tensorboard: TensorboardSection = Field(default_factory=TensorboardSection)
    load_model: LoadModelSection = Field(default_factory=LoadModelSection)
