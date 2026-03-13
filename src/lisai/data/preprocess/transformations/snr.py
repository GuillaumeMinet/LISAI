from __future__ import annotations

import numpy as np


def compute_gt_snr0(stack: np.ndarray, index: int = 0) -> np.ndarray:
    """
    Return the reference SNR frame from a stack.
    """
    return stack[index]


def compute_gt_avg(stack: np.ndarray, n_frames: int | None = None) -> np.ndarray:
    """
    Compute average of first n_frames of stack.
    If n_frames is None, average entire stack.
    """
    if n_frames is None:
        return stack.mean(axis=0)

    return stack[:n_frames].mean(axis=0)
