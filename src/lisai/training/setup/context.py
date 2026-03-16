from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    import torch

    from lisai.config.models import ResolvedExperiment
    from lisai.infra.paths import Paths
    from lisai.runtime.spec import RunSpec


@dataclass
class TrainingContext:
    cfg: ResolvedExperiment
    spec: RunSpec
    paths: Paths

    exp_name: str
    mode: str
    volumetric: bool

    run_dir: Path | None = None
    writer: Any | None = None
    logger: logging.Logger | None = None
    device: torch.device | None = None
    callbacks: list[Any] = field(default_factory=list)
    console_filter: Any | None = None
    file_filter: Any | None = None
    enable_console_logs: Callable[[bool], None] | None = None
    enable_file_logs: Callable[[bool], None] | None = None
