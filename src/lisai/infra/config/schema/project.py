from __future__ import annotations

from typing import Dict

from pydantic import BaseModel, ConfigDict


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


class ProjectConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    project: ProjectMeta
    paths: ProjectPaths
    run_layout: RunLayout
    naming: Naming
