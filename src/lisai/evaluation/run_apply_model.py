import warnings
from pathlib import Path
from typing import Union

import numpy as np
import torch
from tifffile import imread

from lisai.data.utils import center_pad, crop_center
from lisai.evaluation.context import (
    get_model_folder,
    resolve_context_length,
    resolve_tiling_size,
    resolve_upsampling_factor,
)
from lisai.evaluation.inference.normalization import denormalize_pred, normalize_inp
from lisai.evaluation.inference.shape import inverse_make_4d, make_4d
from lisai.evaluation.inference.stack import predict_4d_stack
from lisai.evaluation.io import create_save_folder, resolve_prediction_inputs, save_outputs
from lisai.evaluation.visualization.z_projection import (
    add_colorbar,
    create_color_coded_image,
    enhance_contrast,
)
from lisai.lib.upsamp import inp_generators
from lisai.models.loader import get_model_for_inference


def _default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_apply_model(model_dataset: str,
                model_subfolder: str,
                model_name: str,
                data_path: Path,
                save_folder: str = "default",
                in_place: bool = False,
                epoch_number: int = None,
                best_or_last: str = "best",
                filters: list = ['tiff','tif'],
                skip_if_contain: list = None,
                crop_size: Union[int,tuple] = None,
                keep_original_shape = True,
                tiling_size: int = None,
                stack_selection_idx: int = None,
                timelapse_max:int = None,
                lvae_num_samples:int = None,
                lvae_save_samples:bool = True,
                denormalize_output = True,
                inp_masking:dict = None,
                save_inp: bool = False,
                downsamp: int = None,
                apply_color_code: bool = False,
                color_code_prm: dict = None,
                dark_frame_context_length: bool = False):
    """
    Apply a model to a file or a full folder of files and save model predictions to desired location.
    """
    device = _default_device()
    prediction_prm = locals()
    if color_code_prm is None:
        color_code_prm = {}

    data_path = Path(data_path)

    model_folder = get_model_folder(dataset_name = model_dataset,
                                    subfolder = model_subfolder,
                                    exp_name = model_name)
    
    model,training_cfg,is_lvae = get_model_for_inference(model_folder,device=device,
                                                        best_or_last=best_or_last,
                                                        epoch_number=epoch_number)
    if is_lvae:
        assert lvae_num_samples is not None, ("for LVAE prediction, number of ",
                                              "samples needs to be specified")
    
    data_norm = training_cfg.get("normalization",{}).get("norm_prm")
    clip = False
    if isinstance(data_norm, dict):
        clip = data_norm.get("clip",False)
        if isinstance(clip,bool) and clip is True:
            clip = 0
    model_norm = training_cfg.get("model_norm_prm",None)

    tiling_size = resolve_tiling_size(training_cfg, tiling_size)

    upsamp = resolve_upsampling_factor(training_cfg)
    print(f"Found upsampling factor to be: {upsamp}\n")

    context_length = resolve_context_length(training_cfg)
    if context_length is not None:
        print(f"Found context length to be: {context_length}\n")

    data_path, list_files, name_file = resolve_prediction_inputs(
        data_path,
        filters=filters,
        skip_if_contain=skip_if_contain,
    )
    print(f"Found #{len(list_files)} files.")
    

    # create/define save_folder
    if in_place:
        warnings.warn("arg:`in_place` set to True, input data will be overwitten by predictions")
        if data_path.is_dir():
            save_folder = data_path
        else:
            save_folder = data_path.parent
    else:
        if save_folder == "default":
            #TODO: use a Paths API that set up the default, not hardcoded here
            save_folder = data_path.parent / f"Predict_{model_subfolder}_{model_name}"
        else:
            save_folder = Path(save_folder)
        save_folder = create_save_folder(path = save_folder)

    # save prediction info
    # with open(save_folder/"info.json", 'w') as f:
    #     prediction_prm["data_path"] = str(prediction_prm.get("data_path"))
    #     json.dump(prediction_prm,f, indent=4)
    #     json.dump(training_cfg, f, indent=4)


    # loop over files
    for idx,file in enumerate(list_files):
        print(f"File {idx+1}/{max(1,len(list_files))}: {file}")

        # try:
        file_path = data_path / file

        # load data
        img = imread(file_path)  

        # normalization
        img = normalize_inp(img,clip,data_norm,model_norm)

        # make all data 4d for consistency + optional limitation 
        # of number of frames per timelapses
        img,timelapse,volumetric = make_4d(img,stack_selection_idx,timelapse_max)
        print(img.shape)
        
        # optional cropping
        if crop_size is not None:
            if isinstance(crop_size,int):
                crop_size = (crop_size,crop_size)
            original_size = img.shape[-2:]
            img = crop_center(img,crop_size)
        
        #Optional additional image preparation: masking
        if inp_masking is not None:
            img = inp_generators.generate_masked_inp(img,**inp_masking)

        if downsamp is not None:
            img = img[...,::downsamp,::downsamp]
        

        # prepare empty arrays
        pred_stack, samples_stack = predict_4d_stack(
            model,
            img,
            device=device,
            is_lvae=is_lvae,
            tiling_size=tiling_size,
            lvae_num_samples=lvae_num_samples,
            lvae_save_samples=lvae_save_samples,
            upsamp=upsamp,
            context_length=context_length,
            dark_frame_context_length=dark_frame_context_length,
            verbose=True,
        )
        
        # repad to original size (optional)
        if crop_size is not None and keep_original_shape:
            pad_width = (max(0,original_size[0]-crop_size[0]),
                         max(0,original_size[1]-crop_size[1]))
            pred_stack = center_pad(pred_stack,pad_width)

            if is_lvae and lvae_save_samples and samples_stack is not None:
                samples_stack = center_pad(samples_stack,pad_width)
        
        # denormalization (optional)
        if denormalize_output:
            pred_stack = denormalize_pred(pred_stack,data_norm,model_norm)
            if is_lvae and lvae_save_samples and samples_stack is not None:
                for sample_id in range(samples_stack.shape[0]):
                    samples_stack[sample_id] = denormalize_pred(samples_stack[sample_id],data_norm,model_norm)
        # saving
        pred_stack = inverse_make_4d(pred_stack,volumetric,timelapse,
                                    lvae_samples=False)
        tosave = {"pred": pred_stack.astype(np.float32)}

        # color coding for volumetric data
        if apply_color_code and volumetric:
            try:
                if context_length is not None and not dark_frame_context_length:
                    pred_stack = pred_stack[:,context_length//2:-context_length//2]
                pred_stack_color_coded = create_color_coded_image(pred_stack,
                                                                  colormap=color_code_prm.get("colormap","turbo"),
                                                                  stack_order="ZTYX")
                pred_stack_color_coded = enhance_contrast(pred_stack_color_coded,color_code_prm.get("saturation",0.35))
                if color_code_prm.get("add_colorbar",True):
                    zmax = (pred_stack.shape[0]-1) * color_code_prm.get("zstep",0)
                    pred_stack_color_coded = add_colorbar(pred_stack_color_coded,zmax=zmax)
                tosave["pred_colorCoded"] = pred_stack_color_coded

            except Exception as e:
                warnings.warn(f"Failed to apply color coding: {e}")
        
        if is_lvae and lvae_save_samples and samples_stack is not None:
            samples_stack = inverse_make_4d(samples_stack,volumetric,timelapse,
                                            lvae_samples=True)
            tosave["samples"] = samples_stack.astype(np.float32)
        
        if name_file is None:
            img_name = file.split('.')[0]
        else:
            img_name = name_file.split('.')[0]
        if save_inp:
            tosave["inp"] = img.astype(np.float32)
        save_outputs(tosave,save_folder,img_name,no_suffix=False)
            
        # except Exception as e:
        #     print(e)
