from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

@dataclass(frozen=True)
class InferenceSpec:
    run_dir: Path

    # build
    architecture: str
    parameters: dict[str, Any]

    # LVAE / normalization
    normalization: dict[str, Any]
    model_norm_prm: dict[str, Any] | None = None
    noise_model_name: Optional[str] = None

    # data-derived (needed for LVAE img_shape)
    patch_size: Optional[int] = None
    downsamp_factor: int = 1

    # load
    checkpoint_method: str = "state_dict"
    checkpoint_selector: str = "best"
    checkpoint_epoch: Optional[int] = None
