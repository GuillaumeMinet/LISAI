from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from lisai.models.loader import prepare_model_for_training
from lisai.runtime.spec import RunSpec

if TYPE_CHECKING:
    from lisai.config.models import ResolvedExperiment


def build_model(spec_or_cfg: RunSpec | ResolvedExperiment, device, model_norm_prm, noise_model):
    """
    Transitional entrypoint: accept either a RunSpec or the resolved training config.
    """
    logger = logging.getLogger("lisai")
    spec = spec_or_cfg if isinstance(spec_or_cfg, RunSpec) else RunSpec(spec_or_cfg)

    model, state = prepare_model_for_training(
        spec=spec.model_spec(),
        device=device,
        model_norm_prm=model_norm_prm,
        noise_model=noise_model,
    )

    logger.info(f"Model initialized: {type(model).__name__}")
    return model, state
