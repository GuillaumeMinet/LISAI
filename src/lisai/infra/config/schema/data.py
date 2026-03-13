from __future__ import annotations

from typing import Dict, List

from pydantic import BaseModel, ConfigDict


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    format: Dict[str, Dict[str, str]]
    subfolders: Dict[str, str]
    data_types: Dict[str, List[str]]
    logs: Dict[str, str]
