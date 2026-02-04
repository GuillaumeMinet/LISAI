from pathlib import Path
import json
import os,sys
import torch
from tifffile import imsave
import numpy as np
sys.path.append(os.getcwd())
from lisai.config_project import CFG as config_project
from lisai.lib.utils.get_paths import get_model_name
from lisai.lib.utils.data_utils import get_saving_shape

_cfg_name = config_project.get("cfg_train_filename")

def load_model(model_folder:Path,device,best_or_last:str = "best",
               local:bool = True):
    
    model_folder = Path(model_folder)

    with open(str(model_folder / _cfg_name)) as f:
        CFG_train = json.load(f)
    
    # get load_method
    # NOTE: use of state_dict if possible, otherwise full_model
    if CFG_train.get("saving_prm").get("state_dict",False) is True:
        load_method = "state_dict"
    else:
        load_method = "entire_model"
    
    # get model_file_name
    model_file_name = get_model_name(load_method=load_method,best_or_last=best_or_last)

    # model load
    if load_method == "entire_model":
        model = torch.load(model_folder / model_file_name)
    else:
        pass


def save_outputs(tosave:dict,save_folder:Path,img_name:str,no_suffix=False):
    """
    Utils function to save model outputs.
    Parameters:
    ----------
    tosave: dict
        dictionnary with items to save, name of item in each key
    save_folder: Path
        saving folder location
    img_name: str
        saving file name will be *img_name*_*key*.tif
    no_suffix: bool, default = False
        If only 1 item and no_suffix = True, item only save with img_name.
        If more than 1 item, saving without suffix would overwrite and only 
        last would be saved. 
    """
    for key,item in tosave.items():
        if item is not None:
            if len(tosave) == 1 and no_suffix:
                path = save_folder / f"{img_name}.tif"
            else:
                path = save_folder / f"{img_name}_{key}.tif"
            if key=="pred_colorCoded":
                imsave(path, item, photometric='rgb')
            else:
                shape = get_saving_shape(item)
                imsave(path,item,imagej=True,
                    metadata={"axes": shape})


def make_4d(img:np.array,stack_selection_idx:int=None,timelapse_max:int=None):
    """
    Specific data preparation for "apply_model", to make all data 
    consistently 4d, while dealing with "stack_selection_idx" or "timelapse_max".
    """

    if len(img.shape) == 2:
        img = np.expand_dims(img,axis=(0,1))
        timelapse = False
        volumetric = False

    elif len(img.shape) == 3:
        if stack_selection_idx is not None:
            if isinstance(stack_selection_idx,int):
                img = img[stack_selection_idx:stack_selection_idx+1]
            elif isinstance(stack_selection_idx,list):
                img = img[stack_selection_idx]
        elif timelapse_max is not None:
            img = img[:min(timelapse_max,img.shape[0])]
        img = np.expand_dims(img,axis=0)
        timelapse = True
        volumetric = False

    elif len(img.shape) == 4:
        # print(img.shape)
        if stack_selection_idx is not None:
            if isinstance(stack_selection_idx,int):
                img = img[stack_selection_idx:stack_selection_idx+1]
            elif isinstance(stack_selection_idx,list):
                img = img[stack_selection_idx]
        elif timelapse_max is not None:
            img = img[:min(timelapse_max,img.shape[0])]
        timelapse = True
        volumetric = True
        img = np.transpose(img,(1,0,2,3))
    else:
        raise ValueError (f"Expected data to be 2d, 3d or 4d,"
                            f"but data has shape {img.shape}.")
    
    return img, timelapse,volumetric


def inverse_make_4d(img:np.array,volumetric:bool,timelapse:bool,
                    lvae_samples=False):
    """
    To inverse effect of "make_4d" and restore original image size.
    Expecting img to be [Z,T,H,W], or [Samples,Z,T,H,W] if lvae_samples.
    """
    if not timelapse:
        if lvae_samples:
            img = img[:,0,0]
        else:
            img = img[0,0]
            
    elif not volumetric:
        if lvae_samples:
            img = img[:,0]
        else:
            img = img[0]
    
    return img