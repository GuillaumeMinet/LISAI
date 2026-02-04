import numpy as np
from typing import Union
from tifffile import imsave
import logging
import torch
import math
from scipy.ndimage import gaussian_filter
import sys,os
from typing import Tuple
import warnings
sys.path.append(os.getcwd())

def get_saving_shape(img: np.array):
    """
    Returns correct saving shape depending on img.shape,
    for tifffile saving:
        imsave(path,img,imageJ=True,metada={"axes":shape}")
    """
    
    if len(img.shape) == 2:
        return "YX"
    elif len(img.shape) == 3:
        return "TYX"
    elif len(img.shape) == 4:
        return "TZYX"
    elif len(img.shape) == 5:
        return "TZCYX"


def adjust_img_size(img: Union[np.array, torch.tensor],mltpl_of: int, mode: str):
    """
    Adjust numpy array or torch tensor size so that its size is 
    a multiple of arg: `mltpl_of`, by cropping or padding the tensor, 
    depending on arg:`mode`.
    """

    if mode not in ["crop","pad"]:
        raise ValueError(f"Expected mode to be 'crop' or 'pad', got {mode}")
    
    img_h,img_w = img.shape[-2:]
     
    # Calculate how much to crop or pad
    new_h = (img_h // mltpl_of) * mltpl_of
    new_w = (img_w // mltpl_of) * mltpl_of

    if mode == "crop":
        img_adjusted = crop_center(img,crop_size=(new_h, new_w))

    elif mode == "pad":
        pad_h = mltpl_of - img_h % mltpl_of if img_h % mltpl_of != 0 else 0
        pad_w = mltpl_of - img_w % mltpl_of if img_w % mltpl_of != 0 else 0
        img_adjusted = center_pad(img,pad_size=(pad_h,pad_w))

    return img_adjusted

def adjust_pred_size(img: Union[np.array, torch.tensor],original_size:Tuple[int,int],upsamp:int):
    """
    Adjust img size to match original_size * upsamp with zero-padding if img too small, or cropping if img too large.
    """
    target_size = (original_size[0]*upsamp,original_size[1]*upsamp)
    if target_size[0] < img.shape[-2] or target_size[1] < img.shape[-1]:
        img = crop_center(img,crop_size=target_size)
    elif target_size[0] > img.shape[-2] or target_size[1] > img.shape[-1]:
        img = center_pad(img,target_size=target_size)
    return img



def adjust_for_tiling(tensor:torch.tensor,
                      tiling_size: int,
                      mltpl_of:int,
                      min_tiling_size: int = 100,
                      min_overlap:int = 50):
    """
    Zero-pad `tensor` so that `tensor_pad` is ready for tiling-prediction with overlap. 
    Note that the effective patch size is defined as (tiling_size + overlap).

    The tiling size and the tensor size (with padding) are both adjusted so that the tiling_size 
    is a multiple of the padded tensor. The overlap is defined so that effective patch size is a 
    multiple of arg:`mltpl_of`.

    Inputs
    ------
    tensor: torch.tensor
        tensor to be padded, should be at least 2d  
    tiling_size: int
        desired tiling size, acts as a maximum size for the tiling
    mltpl_of: int
        the effective patch size (tiling_size + overlap) should be a multiple of this value
    min_tiling_size: int, default=100
        minimum tiling size to consider. Should be smaller than tensor size.
    min_overlap: int, default=50
        minimum overlap size to consider.
    """
    
    img_h,img_w = tensor.shape[-2:]
    assert img_h >= min_tiling_size and img_w >= min_tiling_size

    # make sure tiling size is even (makes everything easier)
    if tiling_size % 2 !=0:
        tiling_size = tiling_size-1

    tiling_size_h, pad_h = find_best_tile(img_h, min_tiling_size, tiling_size)
    tiling_size_w, pad_w = find_best_tile(img_w, min_tiling_size, tiling_size)

    # define overlap so that (tile_size + overlap) % arg:`mltpl_of` = 0
    overlap_h = min_overlap + (mltpl_of - (tiling_size_h + min_overlap) % mltpl_of)
    overlap_w = min_overlap + (mltpl_of - (tiling_size_w + min_overlap) % mltpl_of)
    # print(mltpl_of,img_h,img_w,tiling_size_h,tiling_size_w,overlap_h,overlap_w)

    # padd tensor
    tensor_pad = center_pad(tensor,pad_size=(overlap_h+pad_h,overlap_w+pad_w))

    return tensor_pad, (tiling_size_h,tiling_size_w), (overlap_h,overlap_w), (pad_h, pad_w)


def find_best_tile(dim, min_tile_size, max_tile_size,alpha=1, beta=0.3):
    """
    Find the best tile size for a given dimension, minimizing the cost function:
        cost = alpha * total_pad - beta * tile_size
    where total_pad is the total number of pixels padded to make the dimension a multiple of tile_size.

    Inputs
    ------
    dim: int
        dimension to be tiled
    min_tile_size: int
        minimum tile size to consider
    max_tile_size: int
        maximum tile size to consider
    alpha: float, default=1
        weight for the padding cost
    beta: float, default=0.3
        weight for the tile size cost

    Returns
    -------
    best_tile: int
        the best tile size found
    best_total_pad: int
        the total padding applied for the best tile size
    """

    assert min_tile_size <= max_tile_size, "min_tile_size should be <= max_tile_size"

    best_cost = float('inf')

    for tile_size in range(min_tile_size, max_tile_size + 1):
        remainder = dim % tile_size
        if remainder == 0:
            total_pad=0
        else:
            total_pad = tile_size - remainder

        cost = alpha * total_pad - beta * tile_size

        if cost < best_cost:
            best_cost = cost
            best_tile = tile_size
            best_total_pad = total_pad

    return best_tile, best_total_pad



def center_pad(img: Union[np.array,torch.tensor],
               pad_size: Tuple[int,int] = None,
               target_size: Tuple[int,int] = None):
    """
    Center padding with zeros in last 2 directions. If the padding cannot be centered,
    off-centered to top-left (i.e. smallest number of padded pixels are towards the top-left).
    
    Inputs
    ------
    img: numpy array or torch tensor
        image to be padded
    pad_size: tuple, default=None
        defines directly the number of pixels to pad in the 2 directions [pad_h,pad_w] 
        Mutually exclusive with target_size, but one of the 2 has to be defined.
    target_size: tuple, default=None
        defines the desired target size for the padded image as [H,W]
        Mutually exclusive with pad_size, but one of the 2 has to be defined.
    
    Output
    ------
    img: numpy array or torch tensor
        padded image

    """

    if pad_size is None:
        if target_size is None:
            raise ValueError("if pad_size is None, target_size must be defined")
        if not isinstance(target_size,tuple) or len(target_size) != 2:
            raise ValueError ( "target_size should be a tuple: (height,width)")
    
        pad_size = (
            abs(target_size[-2]-img.shape[-2]),
            abs(target_size[-1]-img.shape[-1]),
        )

        if pad_size[0]==0 and pad_size[1]==0:
            return img
    
    else:
        if not isinstance(pad_size,tuple) or len(pad_size) != 2:
            raise ValueError ( "pad_size should be a tuple with: (pad_size_h,pad_size_w)")

    padh = (math.floor(pad_size[0]/2), math.ceil(pad_size[0]/2))
    padw = (math.floor(pad_size[1]/2), math.ceil(pad_size[1]/2))

    # print(pad_size,padh,padw)

    if isinstance(img,np.ndarray):
        if len(img.shape) == 2:
            pad_width = (padh,padw)
        elif len(img.shape) == 3:
            pad_width = ((0,0),padh,padw)
        elif len(img.shape) == 4:
            pad_width = ((0,0),(0,0),padh,padw)
        elif len(img.shape) == 5:
            pad_width = ((0,0),(0,0),(0,0),padh,padw)
        
        img_pad = np.pad(img,pad_width=pad_width,mode='constant', constant_values=0)

    elif isinstance(img,torch.Tensor):
        pad_width = (padw[0],padw[1],padh[0],padh[1]) # 2d torch padding: (left,right,top,bottom)
        img_pad = torch.nn.functional.pad(img,pad_width,mode='constant',value=0)

    return img_pad


def crop_center(img: Union[np.array,torch.tensor],
                crop_size: Union[int,tuple]):
    """
    Center-crops any array or tensor of dim >= 2 according to crop_size.
    If the crop is not centered, it is always off-centered top the top-left.
    """

    if type(crop_size) == tuple:
        crop_h,crop_w = crop_size
    elif type(crop_size) == int:
        crop_h = crop_size
        crop_w = crop_size
    else:
        raise TypeError("crop_size should be tuple or integer.")
    
    img_h,img_w = img.shape[-2::]
    start_h = math.floor(img_h/2-(crop_h/2))
    start_w =  math.floor(img_w/2-(crop_w/2))

    stop_h = start_h+crop_h
    stop_w = start_w+crop_w

    return img[...,start_h:stop_h,start_w:stop_w]


def make_single_4d(img: np.array):
    """
    Make img 4d.
    """
    if len(img.shape) == 2:
        img = np.expand_dims(img,axis=(0,1))

    elif len(img.shape) == 3:
        img = np.expand_dims(img,axis=(0))
    return img


def make_pair_4d(inp:np.array,gt:np.array=None):
    """
    Make pair (inp,gt) of 2d or 3d numpy array 4d.
    If gt is None, keeps it None.
    """

    inp = make_single_4d(inp)
    if gt is not None: 
        gt = make_single_4d(gt)
    
    return inp,gt



def extract_patches(image, patch_size, step=None, max_patches=None):
    """
    Extracts patches in a deterministic manner, for a 2d or 3d image.
    For a 3d, the sliding windows moves only in 2d, but the extracted
    patches have a 3rd dimension.
    Returns an array with one additional dimension on the first axis.
    Example: 
        `img` is 4d with [Snr,Time,H,W]
        => `patches` is 5d with [#patches,Snr,Time,Hpatch,Wpatch] 
        and all patches have been extracted the same along Snr and Time dimensions.
    
    """

    assert len(image.shape) >=2
        
    img_h, img_w = image.shape[-2:]
    if isinstance(patch_size,tuple):
        patch_h, patch_w = patch_size
    else:
        patch_h,patch_w = (patch_size,patch_size)
    
    # Calculate number of patches
    if step is None:
        step=patch_size
    npatch_h = (img_h - patch_h) // step + 1
    npatch_w = (img_w - patch_w) // step + 1
    total_patches =  npatch_h * npatch_w 
    
    # Initialize patches container
    shape = (total_patches, *(image.shape[:-2]), patch_h, patch_w)
    patches = np.empty(shape, dtype=image.dtype)
    
    # extract patches
    patch_idx = 0
    for y in range(0, npatch_h * step, step):
        for x in range(0, npatch_w * step, step):
            # Check if we exceed the bounds of the image
            if y + patch_h <= img_h and x + patch_w <= img_w:
                patch = image[...,y:y+patch_h, x:x+patch_w]
                patches[patch_idx] = patch
                patch_idx += 1
                
            # Stop if we have already extracted the maximum number of patches
            if max_patches and patch_idx >= max_patches:
                return patches[:patch_idx]
    
    return patches


def select_patches(inp_patches:np.array,gt_patches:np.array = None,threshold:float = 0.2,
                   verbose:bool=False,select_on_gt:bool = False):
    """
    Select patches according to threshold. 
    If paired dataset, selection can be done on gt or inp, depending on arg:`select_on_gt`.
    NOTE: selection always done on first frame of axis 1.
    Returns inp_patches and gt_patches (None if unpaired) as 3d numpy array, and # removed patches.
    """

    if isinstance(threshold,bool):
        warnings.warn("threshold should be a float, not a boolean. Using 0.2 as default threshold.")
        threshold = 0.2

    assert len(inp_patches.shape)==4, f"expecting 4d image: [patch,time,h,w], but go shape {inp_patches.shape}"

    len_before = inp_patches.shape[0]

    if gt_patches is not None and select_on_gt:
        leading_patches = gt_patches
    else:
        leading_patches = inp_patches
    
    norm_patches = np.linalg.norm(leading_patches[:,0,...],axis=(1,2))
    norm_patches = norm_patches / norm_patches.max()
    ind_norm = np.where(norm_patches > threshold)[0] #[0] because np.where returns a tuple even if one axis

    len_after = len(ind_norm)
    n_removed = len_before - len_after
    if verbose:
        print(f"Patch selection: {len_before - len_after} out of {len_before} patches removed.")
    
    
    # if (len_before - len_after) / len_before > 0.5:
    #     imsave("kept.tiff",inp_patches[ind_norm,...])
    #     imsave("removed.tiff",inp_patches[ind_norm_2,...])
    
    # if len_before - len_after == 0:
    #     imsave("allkept.tiff",inp_patches[ind_norm,...])

    if gt_patches is not None:
        return inp_patches[ind_norm,...], gt_patches[ind_norm,...], n_removed
    else:
        return inp_patches[ind_norm,...],None,n_removed
    

def bleach_correct_simple_ratio(stack):
    """
    Simple ratio bleach correction.
    Input should be numpy array [T H W]

    """
    logger = logging.getLogger('Bleach_correction')
    time_length = np.shape(stack)[0]
    stack_bc = np.zeros_like(stack)
    int_mean_0 = np.mean(stack[0,...])

    for time in range (time_length):
        ratio = int_mean_0 / np.mean(stack[time,...])
        stack_bc[time,...] = stack[time,...] * ratio
    
    logger.info("Bleach correction finished.")
    return stack_bc


def augment_data(X_train):
    """Augment data by 8-fold with 90 degree rotations and flips. 
    Parameters
    ----------
    X_train: numpy array
        Array of training images.
    """
    X_ = X_train.copy()

    X_train_aug = np.concatenate((X_train, np.rot90(X_, 1, (-2, -1))))
    X_train_aug = np.concatenate((X_train_aug, np.rot90(X_, 2, (-2,-1))))
    X_train_aug = np.concatenate((X_train_aug, np.rot90(X_, 3, (-2,-1))))
    X_train_aug = np.concatenate((X_train_aug, np.flip(X_train_aug, axis=-2)))

    return X_train_aug


def simple_transforms(img,transforms:dict):
    """ Applies transforms to img"""
    
    img = make_single_4d(img)
    for tf,prm in transforms.items():
        if tf == "gauss_blur":
            radius = prm[0]
            sigma = prm[1]
            for p in range(img.shape[0]):
                for ch in range(img.shape[1]):
                    img[p,ch] = gaussian_filter(img[p,ch],sigma=sigma,radius=radius)
        else:
            raise ValueError(f"{tf} transform unknown")
    
    return img

    
