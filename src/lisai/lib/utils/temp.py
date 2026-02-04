import torch
import torch.nn.functional as F

def compute_padding(dim, tile_size):
    remainder = dim % tile_size
    if remainder == 0:
        return tile_size, 0, 0
    pad_total = tile_size - remainder
    pad_before = pad_total // 2
    pad_after = pad_total - pad_before
    return tile_size, pad_before, pad_after

def find_best_tile_for_dim(dim, min_tile_size, max_tile_size):
    best_tile = None
    best_total_pad = float('inf')
    best_pad_before, best_pad_after = 0, 0

    for tile_size in range(min_tile_size, max_tile_size + 1):
        _, pad_before, pad_after = compute_padding(dim, tile_size)
        total_pad = pad_before + pad_after

        is_better = False
        if total_pad < best_total_pad:
            is_better = True
        elif total_pad == best_total_pad and (best_tile is None or tile_size > best_tile):
            is_better = True

        if is_better:
            best_tile = tile_size
            best_total_pad = total_pad
            best_pad_before, best_pad_after = pad_before, pad_after

    return best_tile, best_pad_before, best_pad_after

def adapt_tiling_and_pad_image_1d(tensor, min_tile_size=64, max_tile_size=256):
    """
    Independently optimizes tiling and padding for height and width.
    """
    img_h, img_w = tensor.shape[-2:]

    tile_h, pad_top, pad_bottom = find_best_tile_for_dim(img_h, min_tile_size, max_tile_size)
    tile_w, pad_left, pad_right = find_best_tile_for_dim(img_w, min_tile_size, max_tile_size)

    padding = (pad_left, pad_right, pad_top, pad_bottom)
    padded_tensor = F.pad(tensor, padding, mode='constant', value=0)

    return padded_tensor, tile_h, tile_w, padding
