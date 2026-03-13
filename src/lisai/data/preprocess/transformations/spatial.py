from __future__ import annotations

import numpy as np


def crop_center_2d(img: np.ndarray, crop_size: int) -> np.ndarray:
    """
    Center crop a 2D image to (crop_size, crop_size).
    """
    if img.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape {img.shape}")

    h, w = img.shape
    if crop_size > min(h, w):
        raise ValueError("crop_size larger than image dimensions")

    y0 = (h - crop_size) // 2
    x0 = (w - crop_size) // 2
    return img[y0:y0 + crop_size, x0:x0 + crop_size]


def crop_center_stack(stack: np.ndarray, crop_size: int) -> np.ndarray:
    """
    Center crop a 3D stack (T,Y,X).
    """
    if stack.ndim != 3:
        raise ValueError(f"Expected 3D stack, got shape {stack.shape}")

    return np.stack([crop_center_2d(frame, crop_size) for frame in stack], axis=0)


def register_stack(stack: np.ndarray, reference_index: int = 0):
    """
    Register stack frames to reference frame using pystackreg.
    """
    try:
        from pystackreg import StackReg
    except ImportError as e:
        raise ImportError("register_stack requires pystackreg. Install it to use registration.") from e


    sr = StackReg(StackReg.RIGID_BODY)
    reference = stack[reference_index]
    return sr.register_transform_stack(stack, reference=reference)
