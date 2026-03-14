from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from lisai.models import load_noise_model

if TYPE_CHECKING:
    from lisai.infra.config.schema import ResolvedExperiment
    from lisai.infra.paths import Paths


logger = logging.getLogger("lisai.setup_noise_model")


def _extract_noise_model_name(cfg: ResolvedExperiment) -> str | None:
    noise_model = getattr(cfg, "noise_model", None)
    if isinstance(noise_model, dict):
        return noise_model.get("name")
    if isinstance(noise_model, str):
        return noise_model
    return getattr(noise_model, "name", None)


def _load_noise_model_norm_prm(noise_model_name: str | None, lisai_paths: Paths):
    if not noise_model_name:
        return None

    nm_path = lisai_paths.noise_model_path(noiseModel_name=noise_model_name)
    norm_path = Path(nm_path).parent / "norm_prm.json"

    if not norm_path.exists():
        return None

    with open(norm_path) as f:
        return json.load(f)


def resolve_noise_model_metadata(cfg: ResolvedExperiment, lisai_paths: Paths):
    """
    Resolve Stage A normalization from the noise-model folder when requested.
    """
    if not bool((cfg.normalization or {}).get("load_from_noise_model", False)):
        return None

    noise_model_name = _extract_noise_model_name(cfg)
    if not noise_model_name:
        raise KeyError("Noise model name not found in config")

    data_norm_prm = _load_noise_model_norm_prm(noise_model_name, lisai_paths)
    if data_norm_prm is None:
        raise ValueError("load_from_noise_model=True but norm_prm.json not found next to noise model.")

    logger.info(f"Resolved noise-model normalization from: {noise_model_name}")
    return data_norm_prm


def load_noise_model_object(noise_model_name: str | None, device, lisai_paths: Paths):
    """
    Load the actual noise-model object used by LVAE likelihoods.
    """
    noise_model, _ = load_noise_model(noise_model_name, device, lisai_paths)
    return noise_model