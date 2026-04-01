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


def register_stack(stack: np.ndarray, reference_index: int = 0) -> np.ndarray:
    """
    Register stack frames to reference frame using pystackreg.
    """
    try:
        from pystackreg import StackReg
    except ImportError as e:
        raise ImportError("register_stack requires pystackreg. Install it to use registration.") from e


    if reference_index not in (0, 1):
        raise ValueError(f"Unsupported reference_index={reference_index}. Expected 0 or 1.")

    sr = StackReg(StackReg.RIGID_BODY)

    return sr.register_transform_stack(stack, reference="first")
    # if reference_index == 0:
    #     return sr.register_transform_stack(stack, reference="first")

    # # For reference_index == 1, move frame 0 to the end so original frame 1 is first.
    # shifted = np.roll(stack, shift=-1, axis=0)
    # registered_shifted = sr.register_transform_stack(shifted, reference="first")
    # return np.roll(registered_shifted, shift=1, axis=0)
