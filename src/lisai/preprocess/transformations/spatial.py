from __future__ import annotations

import numpy as np
import math

def crop_center_2d(img: np.ndarray, crop_size: int | tuple) -> np.ndarray:
    """
    Center crop a 2D image to (crop_size, crop_size).
    If the crop is not centered, it is always off-centered top the top-left.
    """
    if type(crop_size) == tuple:
        crop_h,crop_w = crop_size
    elif type(crop_size) == int:
        crop_h = crop_size
        crop_w = crop_size
    else:
        raise TypeError("crop_size should be tuple or integer.")
    
    if img.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape {img.shape}")

    img_h, img_w = img.shape
    if crop_size > min(img_h,img_w):
        raise ValueError("crop_size larger than image dimensions")

    y0 = math.floor(img_h/2-(crop_h/2))
    x0 = math.floor(img_w/2-(crop_w/2))

    return img[y0:y0 + crop_h, x0:x0 + crop_w]


def crop_center_stack(stack: np.ndarray, crop_size: int) -> np.ndarray:
    """
    Center crop a 3D stack (T,Y,X).
    """
    if stack.ndim != 3:
        raise ValueError(f"Expected 3D stack, got shape {stack.shape}")

    return np.stack([crop_center_2d(frame, crop_size) for frame in stack], axis=0)


def register_stack_pystackreg(stack: np.ndarray, reference_index: int = 0) -> np.ndarray:
    """
    Register stack frames to reference frame using pystackreg.
    """
    try:
        from pystackreg import StackReg
    except ImportError as e:
        raise ImportError("register_stack_pystackreg requires pystackreg. Install it to use registration.") from e


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


def register_stack_skimage(
    stack,
    reference_index=0,
    upsample_factor=10,
    interpolation_order=1,
    mode="nearest",
    return_shifts=False,
):
    """
    Register all images in `stack` to the image at `reference_index`
    using skimage.registration.phase_cross_correlation.

    Parameters
    ----------
    stack : np.ndarray
        Array of shape (n_images, height, width).
    reference_index : int
        Index of the reference image, usually the highest-SNR image.
    upsample_factor : int
        Subpixel registration precision. Higher = finer but slower.
        Example: 10 means ~1/10 pixel precision.
    interpolation_order : int
        Interpolation order for scipy.ndimage.shift.
        0 = nearest, 1 = bilinear, 3 = cubic.
    mode : str
        Border handling mode for scipy.ndimage.shift.
    return_shifts : bool
        If True, also return the estimated shifts.

    Returns
    -------
    registered : np.ndarray
        Registered stack with same shape as input.
    shifts : np.ndarray, optional
        Array of shape (n_images, 2), each row = (dy, dx).
        Returned only if return_shifts=True.
    """
    try:
        from scipy.ndimage import shift as ndi_shift
        from skimage.registration import phase_cross_correlation
    except ImportError as e:
        raise ImportError("register_stack_skimage requires scipy.ndimage and skimage.registration") from e
    
    stack = np.asarray(stack)
    if stack.ndim != 3:
        raise ValueError(f"`stack` must have shape (n_images, height, width), got {stack.shape}")

    n_images = stack.shape[0]
    reference = stack[reference_index]
    registered = np.empty_like(stack, dtype=np.float32)
    shifts = np.zeros((n_images, 2), dtype=np.float32)

    for i in range(n_images):
        if i == reference_index:
            registered[i] = stack[i]
            shifts[i] = (0.0, 0.0)
            continue

        shift, error, phasediff = phase_cross_correlation(
            reference_image=reference,
            moving_image=stack[i],
            upsample_factor=upsample_factor,
        )

        dy, dx = shift

        registered[i] = ndi_shift(
            stack[i].astype(np.float32),
            shift=(dy, dx),
            order=interpolation_order,
            mode=mode,
            prefilter=(interpolation_order > 1),
        )
        shifts[i] = (dy, dx)

    if np.issubdtype(stack.dtype, np.integer):
        info = np.iinfo(stack.dtype)
        registered = np.clip(registered, info.min, info.max).astype(stack.dtype)
    else:
        registered = registered.astype(stack.dtype, copy=False)

    if return_shifts:
        return registered, shifts
    return registered