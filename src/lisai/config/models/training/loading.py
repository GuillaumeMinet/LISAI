from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

CheckpointMethod = Literal["state_dict", "full_model"]


class ExperimentLoadModelSection(BaseModel):
    """User-authored settings for loading an existing model into a training run."""

    model_config = ConfigDict(extra="allow", protected_namespaces=())

    canonical_load: bool = Field(
        default=True,
        description="Whether the source run should be resolved through the canonical model directory routing.",
    )
    dataset_name: Optional[str] = Field(
        default=None,
        description="Dataset name of the source run to load from when using canonical loading.",
    )
    subfolder: Optional[str] = Field(
        default=None,
        description="Optional model subfolder used to locate the source run under the models root.",
    )
    exp_name: Optional[str] = Field(
        default=None,
        description="Experiment name of the source run to load from.",
    )
    name: Optional[str] = Field(
        default=None,
        description="Legacy alias for the source experiment name.",
    )
    model_name: Optional[str] = Field(
        default=None,
        description="Legacy alias for the model or run name to load.",
    )
    model_full_path: Optional[str] = Field(
        default=None,
        description="Absolute path to a specific model directory or checkpoint when bypassing canonical routing.",
    )
    load_method: Optional[CheckpointMethod] = Field(
        default=None,
        description="Checkpoint serialization format to load, typically state_dict or full_model.",
    )
    best_or_last: Optional[str] = Field(
        default=None,
        description="Checkpoint selector used when epoch_number is not set, usually best or last.",
    )
    epoch_number: Optional[int] = Field(
        default=None,
        description="Specific checkpoint epoch to load instead of using best_or_last.",
    )


class LoadCheckpoint(BaseModel):
    """Resolved checkpoint selection used internally when loading a prior run."""

    model_config = ConfigDict(extra="allow")

    method: Optional[CheckpointMethod] = Field(
        default=None,
        description="Checkpoint serialization format resolved for loading.",
    )
    selector: Optional[str] = Field(
        default=None,
        description="Checkpoint selector resolved for loading, such as best, last, or epoch.",
    )
    epoch: Optional[int] = Field(
        default=None,
        description="Resolved checkpoint epoch number when a specific epoch is selected.",
    )
    filename: Optional[str] = Field(
        default=None,
        description="Concrete checkpoint filename resolved for loading.",
    )


class LoadModelSection(BaseModel):
    """Runtime model-loading settings after the source run has been resolved."""

    model_config = ConfigDict(extra="allow")

    enabled: bool = Field(
        default=False,
        description="Whether the training run should initialize from an existing checkpoint.",
    )
    source: Optional[str] = Field(
        default=None,
        description="Human-readable source descriptor for the loaded run or checkpoint.",
    )
    run_dir: Optional[str] = Field(
        default=None,
        description="Resolved directory of the source run used for loading.",
    )
    checkpoint: LoadCheckpoint = Field(
        default_factory=LoadCheckpoint,
        description="Resolved checkpoint selection details for the source run.",
    )
