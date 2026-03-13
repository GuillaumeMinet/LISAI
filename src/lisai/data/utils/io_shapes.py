import numpy as np


def get_saving_shape(img: np.ndarray) -> str:
    """
    Returns ImageJ axes string for tifffile saving metadata.
    """
    if len(img.shape) == 2:
        return "YX"
    if len(img.shape) == 3:
        return "TYX"
    if len(img.shape) == 4:
        return "TZYX"
    if len(img.shape) == 5:
        return "TZCYX"
    raise ValueError(f"Unsupported img shape: {img.shape}")
