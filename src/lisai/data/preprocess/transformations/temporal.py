from __future__ import annotations

import numpy as np


def remove_first_frame(stack: np.ndarray) -> np.ndarray:
    """
    Remove first frame of a stack (T,Y,X).
    """
    if stack.ndim != 3:
        raise ValueError("Expected 3D stack")
    return stack[1:]


def bleach_correct_simple_ratio(stack: np.ndarray) -> np.ndarray:
    """
    Simple bleach correction by normalizing each frame
    by the mean intensity of the first frame.
    """
    if stack.ndim != 3:
        raise ValueError("Expected 3D stack")

    ref_mean = stack[0].mean()
    corrected = []
    for frame in stack:
        scale = ref_mean / (frame.mean() + 1e-12)
        corrected.append(frame * scale)

    return np.stack(corrected, axis=0)


def gather_frames(stack: np.ndarray, indices: list[int]) -> np.ndarray:
    """
    Select specific frames from a stack.
    """
    return stack[indices]
