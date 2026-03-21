from __future__ import annotations

from typing import Dict

from pydantic import BaseModel, ConfigDict, Field


class ProjectMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str


class ProjectPaths(BaseModel):
    model_config = ConfigDict(extra="forbid")
    roots: Dict[str, str]
    templates: Dict[str, str]


class RunLayout(BaseModel):
    model_config = ConfigDict(extra="forbid")
    subdirs: Dict[str, str]
    artifacts: Dict[str, str]
    retrain_origin_artifacts: Dict[str, str]


class Naming(BaseModel):
    model_config = ConfigDict(extra="forbid")
    exp_name_format: str
    sample_id: str
    run_dir_index_width: int = Field(default=2, ge=1, le=6)


class RunTracking(BaseModel):
    model_config = ConfigDict(extra="forbid")
    active_heartbeat_timeout_minutes: int = Field(default=10, ge=1)


class QueueResourceClassVRAM(BaseModel):
    model_config = ConfigDict(extra="forbid")

    light: int = Field(default=2000, ge=1)
    medium: int = Field(default=4000, ge=1)
    heavy: int = Field(default=6000, ge=1)


class QueueConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    resource_class_vram_mb: QueueResourceClassVRAM = Field(default_factory=QueueResourceClassVRAM)
    safety_margin_mb: int = Field(default=3000, ge=0)
    poll_seconds: int = Field(default=5, ge=1)


class ProjectConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    project: ProjectMeta
    paths: ProjectPaths
    run_layout: RunLayout
    naming: Naming
    run_tracking: RunTracking = Field(default_factory=RunTracking)
    queue: QueueConfig = Field(default_factory=QueueConfig)
