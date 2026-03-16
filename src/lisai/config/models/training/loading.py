from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

CheckpointMethod = Literal["state_dict", "full_model"]


class ExperimentLoadModelSection(BaseModel):
    model_config = ConfigDict(extra="allow", protected_namespaces=())
    canonical_load: bool = True
    dataset_name: Optional[str] = None
    subfolder: Optional[str] = None
    exp_name: Optional[str] = None
    name: Optional[str] = None
    model_name: Optional[str] = None
    model_full_path: Optional[str] = None
    load_method: Optional[CheckpointMethod] = None
    best_or_last: Optional[str] = None
    epoch_number: Optional[int] = None


class LoadCheckpoint(BaseModel):
    model_config = ConfigDict(extra="allow")
    method: Optional[CheckpointMethod] = None
    selector: Optional[str] = None
    epoch: Optional[int] = None
    filename: Optional[str] = None


class LoadModelSection(BaseModel):
    model_config = ConfigDict(extra="allow")
    enabled: bool = False
    source: Optional[str] = None
    run_dir: Optional[str] = None
    checkpoint: LoadCheckpoint = Field(default_factory=LoadCheckpoint)
