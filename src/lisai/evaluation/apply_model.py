import os,sys
import warnings
import torch
from typing import Union
import json
import numpy as np
from pathlib import Path
from tifffile import imread

sys.path.append(os.getcwd())
from lisai.lib.upsamp import inp_generators
from lisai.lib.utils.data_utils import crop_center,center_pad
from lisai.lib.utils.misc import create_save_folder
from lisai.lib.utils.get_paths import get_model_folder
from lisai.lib.utils.get_model import get_model_for_inference
from lisai.evaluation.helpers.predict import predict
from lisai.evaluation.helpers.eval_utils import save_outputs,make_4d,inverse_make_4d
from lisai.config_project import CFG
from lisai.evaluation.helpers.zProj import create_color_coded_image,enhance_contrast,add_colorbar

_default_tiling_size = CFG.get("default_tiling_size")

# gpu or cpu
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print("Device: ", device)


def apply_model(local:bool,
                model_dataset: str,
                model_subfolder: str,
                model_name: str,
                data_path: Path,
                save_folder: str = "default",
                in_place: str = False,
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
                color_code_prm: dict = {},
                dark_frame_context_length: bool = False):
    """
    Apply a model to a file or a full folder of files and save model predictions to desired location.
    """
    prediction_prm = locals()

    data_path = Path(data_path)
    if isinstance(filters,str):
        filters = [filters]

    model_folder = get_model_folder(local=local,
                                    dataset_name = model_dataset,
                                    subfolder = model_subfolder,
                                    exp_name = model_name)
    
    model,training_cfg,is_lvae = get_model_for_inference(model_folder,device=device,
                                                        best_or_last=best_or_last,
                                                        epoch_number=epoch_number,
                                                        local=local)
    if is_lvae:
        assert lvae_num_samples is not None, ("for LVAE prediction, number of ",
                                              "samples needs to be specified")
    
    data_norm = training_cfg.get("normalization",{}).get("norm_prm")
    if data_norm is not None:
        clip = data_norm.get("clip",False)
        if isinstance(clip,bool) and clip is True:
            clip = 0
    model_norm = training_cfg.get("model_norm_prm",None)

    if tiling_size is None:
        tiling_size = _default_tiling_size.get(training_cfg.get("model_architecture"))

    possible_keys = ["upsamp", "upsampling_factor"]
    upsamp = next((training_cfg.get("model_prm").get(key) for key in possible_keys if training_cfg.get("model_prm").get(key) is not None), None)
    if upsamp is None:
        upsamp = 1
    print(f"Found upsampling factor to be: {upsamp}\n")

    context_length = training_cfg.get("data_prm").get("timelapse_prm",{}).get("context_length",None)
    if context_length is not None:
        print(f"Found context length to be: {context_length}\n")

    # get list of files to apply model on
    if data_path.is_dir():
        toremove=[]
        name_file = None
        list_files = os.listdir(data_path)
        for f in list_files:
            if f.split('.')[-1] not in filters:
                toremove.append(f)
                print(f"Removing {f} because not in filters {filters}")

            if skip_if_contain is not None:
                for skip in skip_if_contain:
                    if skip in f:
                        toremove.append(f)
                        print(f"Removing {f} because contains {skip}")  
                        break  
        for f in toremove:
            list_files.remove(f)

        if len(list_files)>0:
            print(f"Found #{len(list_files)} files.")
        else:
            raise FileNotFoundError(f"No file found in {data_path}")
        
    elif data_path.is_file():
        list_files = [""]
        name_file = data_path.name
    elif data_path.suffix in ['.tif','.tiff']:
        suffix_to_try = '.tif' if data_path.suffix == '.tiff' else '.tiff'
        path_to_try = data_path.with_suffix(suffix_to_try)
        # print(path_to_try)
        if path_to_try.is_file():
            print(f"Found file '{data_path.with_suffix('').name}' with suffix '{suffix_to_try}' instead of '{data_path.suffix}'")
            list_files = [""]
            data_path = path_to_try
            name_file = path_to_try.name
        else:
            raise FileNotFoundError
    else:
        raise FileNotFoundError
    

    # create/define save_folder
    if in_place:
        warnings.warn("arg:`in_place` set to True, input data will be overwitten by predictions")
        if data_path.is_dir():
            save_folder = data_path
        else:
            save_folder = data_path.parent
    else:
        if save_folder == "default":
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
        print(f"File {idx+1}/{max(1,len(list_files)-1)}: {file}")

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
        output_shape = (*img.shape[:-2],img.shape[-2]*upsamp,img.shape[-1]*upsamp)
        # print(output_shape)
        pred_stack = np.empty(shape=output_shape)
        if is_lvae and lvae_save_samples:
            samples_stack = np.empty(shape = (lvae_num_samples,*output_shape))
        
        # loop over z and t axis 
        for z in range(img.shape[0]):
            for t in range(img.shape[1]):
                if context_length is not None:
                    start = t-context_length//2 
                    end = t+context_length//2+1
                    if start < 0 or end > img.shape[1]:
                        if dark_frame_context_length:
                            x = np.zeros((1,context_length,img.shape[-2],img.shape[-1]),dtype=img.dtype)
                            x [:,context_length//2] = img[z,t,...]
                        else:
                            print(f"Skipping frame {t} because not enough context_length")
                            continue
                    else:
                        x = img[z,start:end,...]
                        x = np.expand_dims(x,axis=(0)) # 4d array [1,C=T,H,W]
                else:
                    x = img[z,t,...]
                    x = np.expand_dims(x,axis=(0,1)) # 4d array [1,1,H,W]

                x = torch.from_numpy(x).to(device) # 4d tensor [B,C,H,W]
                # print(x.shape)
                ch_out = 1 if context_length is not None else None               
                outputs = predict(model,x,is_lvae=is_lvae,
                                tiling_size=tiling_size,
                                num_samples=lvae_num_samples,
                                upsamp=upsamp,ch_out=ch_out)
                
                # print(outputs.get("prediction").shape)
                pred_stack[z,t,...] = outputs.get("prediction")
                if is_lvae and lvae_save_samples:
                    samples_stack[:,z,t,...] = outputs.get("samples")
        
        # repad to original size (optional)
        if crop_size is not None and keep_original_shape:
            pad_width = (max(0,original_size[0]-crop_size[0]),
                         max(0,original_size[1]-crop_size[1]))
            pred_stack = center_pad(pred_stack,pad_width)

            if is_lvae and lvae_save_samples:
                samples_stack = center_pad(samples_stack,pad_width)
        
        # denormalization (optional)
        if denormalize_output:
            pred_stack = denormalize_pred(pred_stack,data_norm,model_norm)
            if is_lvae and lvae_save_samples:
                for s in samples_stack:
                    s = denormalize_pred(s,data_norm,model_norm)
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
        
        if is_lvae and lvae_save_samples:
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
            


def normalize_inp(inp,clip,data_norm,model_norm):
    """
    Normalize img for being ready for prediction. 
    """

    if not isinstance(clip,bool):
        inp[inp<clip] = clip

    # inp = (inp - np.mean(inp)) / np.std(inp)

    if data_norm.get("normalize_data"):
        inp = (inp - data_norm.get("avgObs")) / data_norm.get("stdObs")
    if model_norm is not None:
        inp = (inp - model_norm.get("data_mean")) / model_norm.get("data_std") 
    
    return inp


def denormalize_pred(pred,data_norm,model_norm):
    """
    Denormalize output of model.
    """
    if model_norm is not None:
        pred = pred * model_norm.get("data_std") + model_norm.get("data_mean") 
    
    if data_norm.get("normalize_data"):
        pred = pred * data_norm.get("avgObs") + data_norm.get("stdObs")
    
    return pred


if __name__ == "__main__":

    # local = True
    # model_dataset = "Mito_34nm_timelapses"
    # model_subfolder = "Upsampling"
    # model_name = "CL5_Upsamp2_RandomPx_UnetRCAN_rg8_rcab12_red16_CharEdge_alpha005"
    # data_path = r"E:\dl_monalisa\Data\Mito_fast\20250326\Recon_mito\todo"
    # save_folder="default"#r"E:\dl_monalisa\Data\Actin_live_timelapses\denoised\HDN_actin_pred_2"

    # apply_model(local,
    #             model_dataset,
    #             model_subfolder,
    #             model_name,
    #             data_path,
    #             save_folder=save_folder,
    #             lvae_num_samples = 1,
    #             lvae_save_samples = True,
    #             crop_size=None,
    #             tiling_size=300,
    #             timelapse_max=None,
    #             best_or_last="best",
    #             save_inp=False,
    #             skip_if_contain=None,
    #             apply_color_code=True,
    #             color_code_prm={"colormap":"rainbow",
    #                             "saturation":0.75,
    #                             "add_colorbar":True,
    #                             "zstep":0.4},
    #             dark_frame_context_length=True)

    # \\deepltestalab\E\dl_monalisa\Models\Mito_fixed\HDN\Snr1-4-9-14_supervised_betaKL1e-3\evaluation_last
    # E:\dl_monalisa\Models\Actin_fixed_mltplSNR_30nm_2\HDN
    local = True
    model_dataset = "Vim_fixed_mltplSNR_30nm"
    model_subfolder = "HDN"
    model_name = "HDN_single_supervisedAvg_GMMsigN2VAvgbis_KL001"
    data_path = r"E:\dl_monalisa\Data\Vim_bleaching\Monalisa1\data\smallroitest"
    save_folder="default"#r"E:\dl_monalisa\Data\Actin_live_timelapses\denoised\HDN_actin_pred_2"

    apply_model(local,
                model_dataset,
                model_subfolder,
                model_name,
                data_path,
                save_folder=save_folder,
                lvae_num_samples = 100,
                lvae_save_samples = True,
                crop_size=None,
                tiling_size=250,
                timelapse_max=10,
                epoch_number=None,
                best_or_last="last",
                save_inp=False,
                skip_if_contain=None,                
                )