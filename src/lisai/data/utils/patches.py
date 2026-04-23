from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
from tifffile import imwrite

def extract_patches(image: np.ndarray, patch_size, step=None, max_patches=None) -> np.ndarray:
    """
    Deterministic patch extraction for 2d/3d+ arrays, sliding in last 2 dims.
    """
    if image.ndim < 2:
        raise ValueError("image must be at least 2D")

    img_h, img_w = image.shape[-2:]
    if isinstance(patch_size, tuple):
        patch_h, patch_w = patch_size
    else:
        patch_h = patch_w = patch_size

    if step is None:
        step = patch_size

    npatch_h = (img_h - patch_h) // step + 1
    npatch_w = (img_w - patch_w) // step + 1
    total_patches = npatch_h * npatch_w

    shape = (total_patches, *(image.shape[:-2]), patch_h, patch_w)
    patches = np.empty(shape, dtype=image.dtype)

    patch_idx = 0
    for y in range(0, npatch_h * step, step):
        for x in range(0, npatch_w * step, step):
            if y + patch_h <= img_h and x + patch_w <= img_w:
                patches[patch_idx] = image[..., y : y + patch_h, x : x + patch_w]
                patch_idx += 1
            if max_patches and patch_idx >= max_patches:
                return patches[:patch_idx]

    return patches[:patch_idx]


def select_patches(
    inp_patches: np.ndarray,
    gt_patches: Optional[np.ndarray] = None,
    threshold: float = 0.2,
    verbose: bool = False,
    select_on_gt: bool = False,
    save_selected_and_removed: bool = False
):
    """
    Select patches based on norm threshold (computed on frame 0).
    Expects inp_patches shape [patch, time, h, w].
    """
    if isinstance(threshold, bool):
        warnings.warn("threshold should be a float, not a boolean. Using 0.2.")
        threshold = 0.2

    if inp_patches.ndim != 4:
        raise ValueError(f"Expected 4d [patch,time,h,w], got {inp_patches.shape}")

    leading = gt_patches if (gt_patches is not None and select_on_gt) else inp_patches

    norms = np.linalg.norm(leading[:, 0, ...], axis=(1, 2))
    norms = norms / (norms.max() if norms.max() != 0 else 1.0)
    idx = np.where(norms > threshold)[0]

    if save_selected_and_removed:
        removed_idxs = np.where(norms <= threshold)[0]
        removed_patches = inp_patches[removed_idxs, ...]
        imwrite("removed_patches.tiff",removed_patches)
        kept_patches = inp_patches[idx, ...]
        imwrite("kept.tiff", kept_patches)

    if verbose:
        print(f"Patch selection: {inp_patches.shape[0] - len(idx)} removed out of {inp_patches.shape[0]}.")

    if gt_patches is not None:
        return inp_patches[idx, ...], gt_patches[idx, ...], inp_patches.shape[0] - len(idx)
    
    return inp_patches[idx, ...], None, inp_patches.shape[0] - len(idx)
