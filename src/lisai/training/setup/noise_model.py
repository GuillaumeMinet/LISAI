from __future__ import annotations
import torch
import logging
import json
import numpy as np

from pathlib import Path

from lisai.models import load_noise_model

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from lisai.infra.paths import Paths
    from lisai.infra.config.schema import ResolvedExperiment


logger = logging.getLogger("lisai.setup_noise_model")


def prepare_noise_model(cfg: ResolvedExperiment, device, lisaiPaths: Paths):
    """ Load noise model and handles data normalization """
    noise_model_name = cfg.noise_model.name
    if not noise_model_name:
        raise KeyError("Noise model name not found in config")
    
    noise_model, nm_norm_prm = load_noise_model(noise_model_name, device, lisaiPaths)
    
    data_norm_prm = None
    if cfg.normalization or {}.get("load_from_noise_model", False):
        if nm_norm_prm is None:
            raise ValueError("load_from_noise_model=True but norm_prm.json not found next to noise model.")
        else:
            data_norm_prm =  nm_norm_prm
    
    return noise_model, data_norm_prm    