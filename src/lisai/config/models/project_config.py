from __future__ import annotations

from typing import Dict

from pydantic import BaseModel, ConfigDict, Field, field_validator


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


class HDNSafeResumeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    auto_use_safe_checkpoint_on_continue: bool = True
    drop_optimizer_scheduler_state_on_safe_resume: bool = False
    rewind_steps: int = Field(default=1, ge=0)
    lr_scale: float = Field(default=0.2, gt=0.0)
    force_grad_clip_max_norm: float | None = Field(default=None, gt=0.0)

    @field_validator("force_grad_clip_max_norm", mode="before")
    @classmethod
    def _normalize_force_grad_clip_max_norm(cls, value):
        if isinstance(value, str) and value.strip().lower() in {"none", "null", ""}:
            return None
        return value


class RecoveryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hdn_safe_resume: HDNSafeResumeConfig = Field(default_factory=HDNSafeResumeConfig)


class QueueResourceClassVRAM(BaseModel):
    model_config = ConfigDict(extra="forbid")

    light: int = Field(default=2000, ge=1)
    medium: int = Field(default=4000, ge=1)
    heavy: int = Field(default=6000, ge=1)


class QueueConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    resource_class_vram_mb: QueueResourceClassVRAM = Field(default_factory=QueueResourceClassVRAM)
    fixed_margin_pct: float = Field(default=0.20, ge=0.0)
    safety_margin_mb: int = Field(default=3000, ge=0)
    poll_seconds: int = Field(default=5, ge=1)
    paused: bool = False
    max_concurrent_runs_per_gpu: int = Field(default=1, ge=1)


class ProjectConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    project: ProjectMeta
    paths: ProjectPaths
    run_layout: RunLayout
    naming: Naming
    run_tracking: RunTracking = Field(default_factory=RunTracking)
    recovery: RecoveryConfig = Field(default_factory=RecoveryConfig)
    queue: QueueConfig = Field(default_factory=QueueConfig)
