from __future__ import annotations

import torch

from .resize import center_pad


def find_best_tile(dim: int, min_tile_size: int, max_tile_size: int, alpha: float = 1.0, beta: float = 0.3):
    """
    Find best tile size minimizing: cost = alpha*total_pad - beta*tile_size
    """
    if min_tile_size > max_tile_size:
        raise ValueError("min_tile_size should be <= max_tile_size")

    best_cost = float("inf")
    best_tile = min_tile_size
    best_total_pad = 0

    for tile_size in range(min_tile_size, max_tile_size + 1):
        remainder = dim % tile_size
        total_pad = 0 if remainder == 0 else tile_size - remainder
        cost = alpha * total_pad - beta * tile_size

        if cost < best_cost:
            best_cost = cost
            best_tile = tile_size
            best_total_pad = total_pad

    return best_tile, best_total_pad


def adjust_for_tiling(
    tensor: torch.Tensor,
    tiling_size: int,
    mltpl_of: int,
    min_tiling_size: int = 100,
    min_overlap: int = 50,
):
    """
    Zero-pad `tensor` so it's ready for tiling prediction with overlap.
    Returns: tensor_pad, (tile_h,tile_w), (overlap_h,overlap_w), (pad_h,pad_w)
    """
    img_h, img_w = tensor.shape[-2:]
    if img_h < min_tiling_size or img_w < min_tiling_size:
        raise ValueError("tensor smaller than min_tiling_size")

    if tiling_size % 2 != 0:
        tiling_size -= 1

    tile_h, pad_h = find_best_tile(img_h, min_tiling_size, tiling_size)
    tile_w, pad_w = find_best_tile(img_w, min_tiling_size, tiling_size)

    overlap_h = min_overlap + (mltpl_of - (tile_h + min_overlap) % mltpl_of)
    overlap_w = min_overlap + (mltpl_of - (tile_w + min_overlap) % mltpl_of)

    tensor_pad = center_pad(tensor, pad_size=(overlap_h + pad_h, overlap_w + pad_w))
    return tensor_pad, (tile_h, tile_w), (overlap_h, overlap_w), (pad_h, pad_w)
