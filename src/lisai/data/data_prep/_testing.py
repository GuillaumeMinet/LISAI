from torchvision import transforms
import numpy as np
from tifffile import imread,imsave
import glob
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import sys
import os


CFG = {

    # Data configuration
    "data_parameters": {
        "dataset_name" : 'Vim_fixed_mltplSNR_30nm',
        "paired": False,
        "gt": "gt_avg",
        "inp": False,
        "data_types":{
            "gt": "recon",
            "inp": "recon"
        },
        "crop_size": 96,
        "augmentation": {
            "hvFlips_rot90": True
        },
        "patch_thresh": 0.2, # false or float [0,1]
        "npatch": 50,
        "clip_neg": {
            "gt": True,
            "inp": True,
        },
        "gt_transform": {
            "gauss_blur": None,
        },
        "inp_transform": {
            "gauss_blur": None,
        },
        
        "downsamp_inp": True,
        "downsamp_prm": {
            "downsamp_factor": 2,
            "downsamp_method": 'real', #'real', 'random', 'blur'
            "sampling_strategy": np.array([[0,0,1,1],[0,1,0,1]]),
       },

       "mltpl_snr_prm":{
           "snr_idx": None,
       },

       "timelapse_prm":{
            "context_length": 1,
            "cumul_time_list": True,
            "frame_selection": None,
       },

       "norm_prm": None,
        "artifical_movement": True,
        "artifical_movement_prm":{
           "movement_type": "translation",
           "translation_prm":{
               "speed": 3,
               "direction": "h+v+",
               "nFrames": 7,
               "dynamic_direction": True,
               "variable_speed": False,
           }
       }
    }
}

data_dir = Path(r"\\deepltestalab\E\dl_monalisa\Data\Vim_fixed_mltplSNR_30nm\preprocess\recon")
sys.path.append(r"c:/Users/guillaume.minet/Documents/GitHub/dl_monalisa/src/")
from dataset_classes.monalisa_dataset import monalisa_dataset as Dataset

data_parameters = CFG["data_parameters"]
train_dataset = Dataset(data_dir=data_dir,
                        split = "train",
                        for_training = True,
                        **data_parameters)


idx = 0
inp,gt,_= train_dataset [idx]
# inp1,inp2 = inp
print(inp.shape)
print(gt.shape)
patch_idx = 8
imsave("inp.tif",inp[patch_idx].numpy())
# imsave("inp1.tif",inp1[patch_idx].numpy())
# imsave("inp2.tif",inp1[patch_idx].numpy())
imsave("gt.tif",gt[patch_idx].numpy())

"""
inp_img = inp[patch_idx,0].cpu().numpy()
gt_img = gt[patch_idx,0].cpu().numpy()

plt.figure(figsize=(10,10))
plt.subplot(1,2,1)
plt.imshow(inp_img,cmap='gray')
plt.title("Input")
plt.subplot(1,2,2)
plt.imshow(gt_img,cmap='gray')
plt.title("GT")
plt.show()"""