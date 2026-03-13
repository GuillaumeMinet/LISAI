# lisai/data/preprocess/config.py
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PreprocessConfig(BaseModel):
    # Decide if you want strict top-level keys:
    model_config = ConfigDict(extra="forbid")  # or "allow" if you prefer

    dataset_name: str
    pipeline: str
    data_type: str
    fmt: str
    pipeline_cfg: dict[str, Any] = Field(default_factory=dict)