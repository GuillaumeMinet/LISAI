"""High-level entrypoint for applying a saved model to arbitrary input files.

This module orchestrates the evaluation pipeline for prediction-only usage:
resolve the saved run, initialize the inference runtime, prepare input files,
run stack inference, and save outputs.
"""

import warnings
from pathlib import Path
from typing import Union

import numpy as np
from tifffile import imread

from lisai.data.utils import center_pad, crop_center
from lisai.evaluation.defaults import UNSET, UnsetType, resolve_apply_options
from lisai.evaluation.inference.normalization import denormalize_pred, normalize_inp
from lisai.evaluation.inference.shape import inverse_make_4d, make_4d
from lisai.evaluation.inference.stack import predict_4d_stack
from lisai.evaluation.io import create_save_folder, resolve_prediction_inputs, save_outputs
from lisai.evaluation.runtime import initialize_runtime
from lisai.evaluation.saved_run import load_saved_run, resolve_run_dir
from lisai.evaluation.visualization.z_projection import (
    add_colorbar,
    create_color_coded_image,
    enhance_contrast,
)


def run_apply_model(model_dataset: str,
                model_subfolder: str,
                model_name: str,
                data_path: Path,
                save_folder: str | Path | None | UnsetType = UNSET,
                in_place: bool | UnsetType = UNSET,
                epoch_number: int | None | UnsetType = UNSET,
                best_or_last: str | UnsetType = UNSET,
                filters: list[str] | str | UnsetType = UNSET,
                skip_if_contain: list[str] | None | UnsetType = UNSET,
                crop_size: Union[int, tuple[int, int], None, UnsetType] = UNSET,
                keep_original_shape: bool | UnsetType = UNSET,
                tiling_size: int | None | UnsetType = UNSET,
                stack_selection_idx: int | None | UnsetType = UNSET,
                timelapse_max: int | None | UnsetType = UNSET,
                lvae_num_samples: int | None | UnsetType = UNSET,
                lvae_save_samples: bool | UnsetType = UNSET,
                denormalize_output: bool | UnsetType = UNSET,
                save_inp: bool | UnsetType = UNSET,
                downsamp: int | None | UnsetType = UNSET,
                apply_color_code: bool | UnsetType = UNSET,
                color_code_prm: dict | None | UnsetType = UNSET,
                dark_frame_context_length: bool | UnsetType = UNSET,
                config: str | Path | None = None):
    """Apply a saved model checkpoint to one file or a directory of files.

    Any omitted optional argument is resolved from `configs/inference/defaults.yml`
    or from the named config passed via `config`.
    """
    options = resolve_apply_options(
        config=config,
        save_folder=save_folder,
        in_place=in_place,
        epoch_number=epoch_number,
        best_or_last=best_or_last,
        filters=filters,
        skip_if_contain=skip_if_contain,
        crop_size=crop_size,
        keep_original_shape=keep_original_shape,
        tiling_size=tiling_size,
        stack_selection_idx=stack_selection_idx,
        timelapse_max=timelapse_max,
        lvae_num_samples=lvae_num_samples,
        lvae_save_samples=lvae_save_samples,
        denormalize_output=denormalize_output,
        save_inp=save_inp,
        downsamp=downsamp,
        apply_color_code=apply_color_code,
        color_code_prm=color_code_prm,
        dark_frame_context_length=dark_frame_context_length,
    )
    color_code_prm = options["color_code_prm"] or {}

    data_path = Path(data_path)
    run_dir = resolve_run_dir(dataset_name=model_dataset, subfolder=model_subfolder, exp_name=model_name)
    saved_run = load_saved_run(run_dir)
    runtime = initialize_runtime(
        saved_run=saved_run,
        best_or_last=options["best_or_last"],
        epoch_number=options["epoch_number"],
        tiling_size=options["tiling_size"],
    )
    if saved_run.is_lvae:
        assert options["lvae_num_samples"] is not None, (
            "for LVAE prediction, number of samples needs to be specified"
        )

    data_norm = saved_run.data_norm_prm
    clip = False
    if isinstance(data_norm, dict):
        clip = data_norm.get("clip", False)
        if isinstance(clip, bool) and clip is True:
            clip = 0
    model_norm = saved_run.model_norm_prm

    tiling_size = runtime.tiling_size
    upsamp = saved_run.upsampling_factor
    print(f"Found upsampling factor to be: {upsamp}\n")

    context_length = saved_run.context_length
    if context_length is not None:
        print(f"Found context length to be: {context_length}\n")

    data_path, list_files, name_file = resolve_prediction_inputs(
        data_path,
        filters=options["filters"],
        skip_if_contain=options["skip_if_contain"],
    )
    print(f"Found #{len(list_files)} files.")

    if options["in_place"]:
        warnings.warn("arg:`in_place` set to True, input data will be overwitten by predictions")
        if data_path.is_dir():
            save_folder = data_path
        else:
            save_folder = data_path.parent
    else:
        if options["save_folder"] == "default":
            save_folder = data_path.parent / f"Predict_{model_subfolder}_{model_name}"
        else:
            save_folder = Path(options["save_folder"])
        save_folder = create_save_folder(path=save_folder)

    for idx, file in enumerate(list_files):
        print(f"File {idx+1}/{max(1, len(list_files))}: {file}")

        file_path = data_path / file
        img = imread(file_path)
        img = normalize_inp(img, clip, data_norm, model_norm)
        img, timelapse, volumetric = make_4d(img, options["stack_selection_idx"], options["timelapse_max"])
        print(img.shape)

        crop_size = options["crop_size"]
        if crop_size is not None:
            if isinstance(crop_size, int):
                crop_size = (crop_size, crop_size)
            original_size = img.shape[-2:]
            img = crop_center(img, crop_size)

        if options["downsamp"] is not None:
            img = img[..., :: options["downsamp"], :: options["downsamp"]]

        pred_stack, samples_stack = predict_4d_stack(
            runtime.model,
            img,
            device=runtime.device,
            is_lvae=saved_run.is_lvae,
            tiling_size=tiling_size,
            lvae_num_samples=options["lvae_num_samples"],
            lvae_save_samples=options["lvae_save_samples"],
            upsamp=upsamp,
            context_length=context_length,
            dark_frame_context_length=options["dark_frame_context_length"],
            verbose=True,
        )

        if crop_size is not None and options["keep_original_shape"]:
            pad_width = (
                max(0, original_size[0] - crop_size[0]),
                max(0, original_size[1] - crop_size[1]),
            )
            pred_stack = center_pad(pred_stack, pad_width)

            if saved_run.is_lvae and options["lvae_save_samples"] and samples_stack is not None:
                samples_stack = center_pad(samples_stack, pad_width)

        if options["denormalize_output"]:
            pred_stack = denormalize_pred(pred_stack, data_norm, model_norm)
            if saved_run.is_lvae and options["lvae_save_samples"] and samples_stack is not None:
                for sample_id in range(samples_stack.shape[0]):
                    samples_stack[sample_id] = denormalize_pred(samples_stack[sample_id], data_norm, model_norm)
        pred_stack = inverse_make_4d(pred_stack, volumetric, timelapse, lvae_samples=False)
        tosave = {"pred": pred_stack.astype(np.float32)}

        if options["apply_color_code"] and volumetric:
            try:
                if context_length is not None and not options["dark_frame_context_length"]:
                    pred_stack = pred_stack[:, context_length // 2 : -context_length // 2]
                pred_stack_color_coded = create_color_coded_image(
                    pred_stack,
                    colormap=color_code_prm.get("colormap", "turbo"),
                    stack_order="ZTYX",
                )
                pred_stack_color_coded = enhance_contrast(
                    pred_stack_color_coded,
                    color_code_prm.get("saturation", 0.35),
                )
                if color_code_prm.get("add_colorbar", True):
                    zmax = (pred_stack.shape[0] - 1) * color_code_prm.get("zstep", 0)
                    pred_stack_color_coded = add_colorbar(pred_stack_color_coded, zmax=zmax)
                tosave["pred_colorCoded"] = pred_stack_color_coded

            except Exception as e:
                warnings.warn(f"Failed to apply color coding: {e}")

        if saved_run.is_lvae and options["lvae_save_samples"] and samples_stack is not None:
            samples_stack = inverse_make_4d(samples_stack, volumetric, timelapse, lvae_samples=True)
            tosave["samples"] = samples_stack.astype(np.float32)

        if name_file is None:
            img_name = file.split('.')[0]
        else:
            img_name = name_file.split('.')[0]
        if options["save_inp"]:
            tosave["inp"] = img.astype(np.float32)

        save_outputs(tosave, save_folder, img_name)
