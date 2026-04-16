from __future__ import annotations

import numpy as np
import torch

from lisai.evaluation.inference.engine import predict


def infer_batch(
    model: torch.nn.Module,
    x: torch.Tensor,
    *,
    is_lvae: bool,
    tiling_size: int | None,
    num_samples: int | None,
    upsamp: int,
    ch_out: int | None,
):
    return predict(
        model,
        x,
        is_lvae=is_lvae,
        tiling_size=tiling_size,
        num_samples=num_samples,
        upsamp=upsamp,
        ch_out=ch_out,
    )


def _build_temporal_input(
    img: np.ndarray,
    *,
    z: int,
    t: int,
    context_length: int | None,
    dark_frame_context_length: bool,
    verbose: bool = False,
) -> np.ndarray | None:
    if context_length is None:
        x = img[z, t, ...]
        return np.expand_dims(x, axis=(0, 1))  # [B=1, C=1, H, W]

    start = t - context_length // 2
    end = t + context_length // 2 + 1
    if start < 0 or end > img.shape[1]:
        if dark_frame_context_length:
            x = np.zeros((1, context_length, img.shape[-2], img.shape[-1]), dtype=img.dtype)
            x[:, context_length // 2] = img[z, t, ...]
            return x
        if verbose:
            print(f"Skipping frame {t} because not enough context_length")
        return None

    x = img[z, start:end, ...]
    return np.expand_dims(x, axis=0)  # [B=1, C=T, H, W]


def predict_4d_stack(
    model: torch.nn.Module,
    img: np.ndarray,
    *,
    timelapse: bool,
    ch_out: int | None,
    device: torch.device,
    is_lvae: bool,
    tiling_size: int | None,
    lvae_num_samples: int | None,
    lvae_save_samples: bool,
    upsamp: int,
    context_length: int | None,
    dark_frame_context_length: bool,
    verbose: bool = False,
):
    if timelapse:
        output_shape = (*img.shape[:-2], img.shape[-2] * upsamp, img.shape[-1] * upsamp)
    else:
        # Non-timelapse inputs use axis 1 as channels, not time.
        output_shape = (img.shape[0], 1, img.shape[-2] * upsamp, img.shape[-1] * upsamp)
    pred_stack = np.empty(shape=output_shape)
    samples_stack = None
    if is_lvae and lvae_save_samples:
        samples_stack = np.empty(shape=(lvae_num_samples, *output_shape))

    for z in range(img.shape[0]):
        t_iter = range(img.shape[1]) if timelapse else range(1)
        for t in t_iter:
            if timelapse:
                x_np = _build_temporal_input(
                    img,
                    z=z,
                    t=t,
                    context_length=context_length,
                    dark_frame_context_length=dark_frame_context_length,
                    verbose=verbose,
                )
            else:
                x_np = np.expand_dims(img[z, ...], axis=0)  # [B=1, C, H, W]
            if x_np is None:
                continue

            x = torch.from_numpy(x_np).to(device)
            resolved_ch_out = ch_out
            if resolved_ch_out is None and context_length is not None:
                resolved_ch_out = 1
            outputs = infer_batch(
                model,
                x,
                is_lvae=is_lvae,
                tiling_size=tiling_size,
                num_samples=lvae_num_samples,
                upsamp=upsamp,
                ch_out=resolved_ch_out,
            )

            pred_stack[z, t, ...] = outputs.get("prediction")
            if is_lvae and lvae_save_samples and samples_stack is not None:
                samples_stack[:, z, t, ...] = outputs.get("samples")

    return pred_stack, samples_stack
