from __future__ import annotations

import logging

import numpy as np
from scipy.ndimage import gaussian_filter


def make_single_4d(img: np.ndarray) -> np.ndarray:
    """
    Make img 4d [snr,time,h,w] style.
    """
    if img.ndim == 2:
        return np.expand_dims(img, axis=(0, 1))
    if img.ndim == 3:
        return np.expand_dims(img, axis=0)
    return img


def make_pair_4d(inp: np.ndarray, gt: np.ndarray | None = None):
    """
    Make (inp, gt) 4d. Keeps gt None if unpaired.
    """
    inp = make_single_4d(inp)
    if gt is not None:
        gt = make_single_4d(gt)
    return inp, gt


def augment_data(x: np.ndarray) -> np.ndarray:
    """
    8-fold augmentation: 90° rotations and flips on last 2 axes.
    """
    x_ = x.copy()
    out = np.concatenate((x, np.rot90(x_, 1, (-2, -1))))
    out = np.concatenate((out, np.rot90(x_, 2, (-2, -1))))
    out = np.concatenate((out, np.rot90(x_, 3, (-2, -1))))
    out = np.concatenate((out, np.flip(out, axis=-2)))
    return out


def simple_transforms(img: np.ndarray, transforms: dict):
    """
    Applies simple transforms to img.
    """
    img = make_single_4d(img)
    for tf, prm in transforms.items():
        if tf == "gauss_blur":
            radius = prm[0]
            sigma = prm[1]
            for p in range(img.shape[0]):
                for ch in range(img.shape[1]):
                    img[p, ch] = gaussian_filter(img[p, ch], sigma=sigma, radius=radius)
        else:
            raise ValueError(f"{tf} transform unknown")
    return img


def bleach_correct_simple_ratio(stack: np.ndarray) -> np.ndarray:
    """
    Simple ratio bleach correction. Input: [T, H, W]
    """
    logger = logging.getLogger("Bleach_correction")
    time_length = stack.shape[0]
    out = np.zeros_like(stack)
    int_mean_0 = float(np.mean(stack[0, ...]))

    for t in range(time_length):
        denom = float(np.mean(stack[t, ...]))
        ratio = int_mean_0 / denom if denom != 0 else 1.0
        out[t, ...] = stack[t, ...] * ratio

    logger.info("Bleach correction finished.")
    return out
