
from __future__ import annotations
import torch
import logging
import json
import numpy as np

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from lisai.infra.paths import Paths

logger = logging.getLogger("lisai.noise_model")

def load_noise_model(noise_model_name: str | None, device: torch.device, lisaiPaths: Paths):
    if not noise_model_name:
        return None, None

    # paths = Paths(settings)
    nm_path = lisaiPaths.noise_model_path(noiseModel_name=noise_model_name)

    if not nm_path.exists():
        raise FileNotFoundError(f"Noise model not found: {nm_path}")

    from lisai.lib.hdn.gaussianMixtureNoiseModel import GaussianMixtureNoiseModel

    nm_params = np.load(nm_path)
    noise_model = GaussianMixtureNoiseModel(params=nm_params, device=device)
    logger.info(f"Loaded noise GMM: {noise_model_name}")

    norm_path = lisaiPaths.noise_model_norm_prm_path(noiseModel_name=noise_model_name)
    nm_norm_prm = None
    if norm_path.exists():
        with open(norm_path) as f:
            nm_norm_prm = json.load(f)

    return noise_model, nm_norm_prm
