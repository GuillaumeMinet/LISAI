from torch.utils.data import TensorDataset,DataLoader
from pathlib import Path
import os,sys,glob,warnings
from tifffile import imread,imsave
import json
import numpy as np
import torch

from lisai.lib.upsamp.inp_generators import generate_masked_inp, generate_downsamp_inp
from lisai.lib.utils.data_utils import (crop_center,extract_patches,select_patches,make_pair_4d,augment_data,simple_transforms)
from lisai.lib.upsamp.artificial_movement import apply_movement
from lisai.lib.utils.get_paths import get_dataset_path
from lisai.lib.utils.logger_utils import CustomStreamHandler

import lisai.config_data as config_data
jsonfile_path = Path(os.getcwd()) / config_data.CFG["datasets_json"]
with open(jsonfile_path, 'r') as file:
    datasets = json.load(file)

import logging
logger = logging.getLogger("data_prep")
logger.addHandler(CustomStreamHandler())


def make_test_loader(**kwargs):
    """
    Key-worded function that makes the loaders for training and validation.
    """

    list_datasets,_,_ = prep_data(for_training = False,**kwargs)
    test_set = TensorDataset(*list_datasets[0])
    test_loader = DataLoader(test_set,batch_size=1,shuffle=False)

    return test_loader


def make_training_loaders(**kwargs):
    """
    Key-worded function that makes the loaders for training and validation.
    """

    prep_before = kwargs.get("prep_before")
    assert prep_before is not None, "Data preparation missing argument `prep_before` (bool)"
    assert isinstance(prep_before,bool), "Expected prep_before to be a boolean"
    if prep_before:
        list_datasets,model_norm_prm,patch_info = prep_data(for_training = True,**kwargs)
        train_set = TensorDataset(*list_datasets[0])
        val_set = TensorDataset(*list_datasets[1]) 
    else:
        raise NotImplementedError() #TODO

    train_loader = DataLoader(train_set,batch_size=kwargs.get("batch_size"),
                              shuffle=True)
    val_loader = DataLoader(val_set,batch_size=kwargs.get("batch_size"),
                            shuffle=False)

    return train_loader,val_loader,model_norm_prm,patch_info


def prep_data(**kwargs):
    """
    Key-worded function that prepares data for training or evaluation,depending
    on arg:`for_trainining`(bool). 
    Note that `data_format` is searched in order: in the "datasets.json" file, then
    in `kwargs`, and finally assumed "single" if not found.
    For training, if dataset not already split in training/validation, an automatic
    split is done with proportion 0.85-0.15.

    Returns:
    --------
    list_datasets: list
        list of datasets: [(inp_train,gt_train),(inp_val,gt_val)] 
        if training, or [(inp_test,gt_test)] if evaluation.

    model_norm: dict
        normalization parameters of the training dataset.
        None if evaluation.
    
    patch_info: dict
        info about #patches for train and validation splits
        None if evaluation.

    """

    for_training = kwargs.get("for_training")
    if for_training is None:
        raise ValueError("boolean arg:`for_training` needs to be specified.")
    assert isinstance(for_training,bool)

    dataset_name = kwargs.get("dataset_name")
    dataset_info = datasets.get(dataset_name)
    if dataset_info is None:
        data_format = kwargs.get("data_format")
        if data_format is None:
            data_format = "timelapse"
            warnings.warn("Data format not specified, put to 'single' by default.")     
    else:
        data_format = dataset_info.get("data_format")

    if kwargs.get("data_dir") is None: 
        kwargs["data_dir"] = get_dataset_path(**kwargs)

    paired = kwargs.get("paired")
    norm_prm = kwargs.get("norm_prm")
    if paired: assert kwargs.get("gt") is not None, "paired dataset necessitates a ground-truth"

    # load dataset(s)
    list_datasets, patch_info = load_full_datasets(data_format=data_format,**kwargs)

    if kwargs.get("masking") is not None or kwargs.get("downsampling") is not None:
         # NOTE: by default, we consider that upsampling-like ==> supervised training.
         # UNLESS we find in the masking/downsampling parameters a key "supervised_training" 
         # set to False.
         # NOTE: If supervised_training and dataset not originally paired, we make it paired:
         # "new inp" = transformed(inp) and gt = "original inp" => "paired" will be updated.
        list_datasets,paired = apply_inp_transformations(list_datasets,**kwargs)
    
    # additional transforms
    apply_additional_transforms(list_datasets,kwargs.get("inp_transform"),kwargs.get("gt_transform"))

    # get final normalization parameters NOTE: for training,
    # we normalize only if not already normalized in data_loading
    # for inference, will be normalized by model_norm_prm if found.
    if for_training:
        # norm_prm = kwargs.get("norm_prm")
        model_norm_prm = calculate_dataset_normalization(*list_datasets)
    
        # if norm_prm is None or not norm_prm.get("normalize_data",False):
        #     model_norm_prm = calculate_dataset_normalization(*list_datasets)
        # else: 
        #     model_norm_prm = {"data_mean": 0,"data_std": 1,
        #                       "data_mean_gt": 0 if paired else None,
        #                       "data_std_gt": 1 if paired else None}  
    else:
        model_norm_prm = kwargs.get("model_norm_prm")
    
    
    list_datasets = apply_normalization(list_datasets,model_norm_prm)

    list_datasets = make_tensor(list_datasets)

    return list_datasets,model_norm_prm,patch_info



def load_full_datasets(data_dir:Path,inp:str,gt:str,paired:bool,
                       data_format:str,for_training:bool,
                       already_split:bool,**kwargs):
    """
    Loads all datasets: train and validation if arg:`for_training` is True, 
    only the test dataset otherwise.
    """
   
    if for_training:
        if already_split:
            # train split
            inp_path = data_dir / inp / "train"
            gt_path =  data_dir / gt / "train" if paired else None
            inp_train,gt_train = load_all_data(inp_path,gt_path,data_format,
                                           make_patches=True,**kwargs)
            
            # val split
            inp_path = data_dir / inp / "val"
            gt_path =  data_dir / gt / "val" if paired else None
            inp_val,gt_val = load_all_data(inp_path,gt_path,data_format,val_split=True,
                                       make_patches=True,**kwargs)
            
        else:
            inp_path = data_dir / inp
            gt_path = data_dir / gt if gt is not None else None
            inp_data,gt_data = load_all_data(inp_path,gt_path,make_patches=True,**kwargs)

            inp_train = inp_data[:int(0.85*inp_data.shape[0])]
            inp_val = inp_data[int(0.85*inp_data.shape[0]):]

            gt_train = gt_data[:int(0.85*gt_data.shape[0])] if paired else None
            gt_val = gt_data[int(0.85*gt_data.shape[0]):] if paired else None

        # train dataset augmentation (optional)
        if kwargs.get("augmentation",False):
            inp_train = augment_data(inp_train)
            if paired:
                gt_train = augment_data(gt_train)

        train_dataset = (inp_train,gt_train)
        val_dataset = (inp_val,gt_val)
        list_datasets = [train_dataset,val_dataset]

        patch_info = {
            "train_patch": inp_train.shape,
            "val_patch": inp_val.shape,
        }
        logger.info(f"Training patches: {inp_train.shape}, Validation patches: {inp_val.shape}")

    else:
        split = kwargs.get("split","test")
        inp_path = data_dir / inp / split
        gt_path =  data_dir / gt / split if gt is not None else None
        inp_test,gt_test = load_all_data(inp_path,gt_path,data_format,make_patches=False,**kwargs)
        test_dataset = (inp_test,gt_test)
        list_datasets = [test_dataset]
        patch_info = None
    
    return list_datasets,patch_info

def apply_inp_transformations(list_datasets:list,**kwargs):
    """
    Apply masking or downsampling transformation for all
    dataset listed in arg:`list_datasets`.
    """
    for_training = kwargs.get("for_training",True)
    if kwargs.get("masking") is not None:
        masking = True
        downsampling = False
        transform_prm = kwargs.get("masking")
    elif kwargs.get("downsampling") is not None:
        masking = False
        downsampling = True
        transform_prm = kwargs.get("downsampling")
    
    if transform_prm.get("supervised_training",True):
        paired = True
    else:
        paired = False

    for i,dataset in enumerate(list_datasets):
        inp,gt = dataset
        if gt is None and paired:
            gt = inp.copy()
            if gt.shape[1]>1 and not kwargs.get("volumetric",False):
                idx = gt.shape[1] // 2
                gt = gt[:,idx:idx+1,...]
        
        if masking:
            if gt is not None and gt.shape[-2:] != inp.shape[-2:]: 
                downsampled_inp = True
                downsamp_factor = gt.shape[-1] // inp.shape[-1]
            else:
                downsampled_inp = False
                downsamp_factor = None
            inp = generate_masked_inp(inp,transform_prm,downsampled_inp,
                                      downsamp_factor)

        elif downsampling:
            if not for_training: # enforce deterministic downsampling for all inferences
                method = transform_prm.get("downsamp_method") 
                if method == "random":
                    transform_prm["downsamp_method"] = "real"
                elif method == "multiple":
                    transform_prm["multiple_prm"]["random"] = False 
            
            inp,_ = generate_downsamp_inp(inp,transform_prm)
    
        list_datasets[i] = (inp,gt)

    return list_datasets, paired



def make_tensor(list_datasets):
    """
    Transforms all dataset found in list_datasets
    to torch.Tensor. If not paired, gt is filled 
    with NaN.
    """
    
    for i,dataset in enumerate(list_datasets):
        inp,gt = dataset
        inp = torch.from_numpy(inp).to(torch.float32)
        if gt is not None:
            gt = torch.from_numpy(gt).to(torch.float32)
        else:
            gt = torch.zeros(inp.shape[0],1,1,1).fill_(float('nan'))
        list_datasets[i] = (inp,gt)
    
    return list_datasets


def apply_normalization(list_datasets,model_norm_prm = None):
    """
    Apply normalization parameters.
    """
    
    if model_norm_prm is None:
        return list_datasets
    
    for i,(inp,gt) in enumerate(list_datasets):

        data_mean = model_norm_prm.get("data_mean")
        data_std = model_norm_prm.get("data_std")
        inp = (inp - data_mean) / data_std

        if gt is not None:
            data_mean_gt = model_norm_prm.get("data_mean_gt")
            data_std_gt = model_norm_prm.get("data_std_gt")
            gt = (gt - data_mean_gt) / data_std_gt
        
        list_datasets[i] = (inp,gt)

    return list_datasets

def calculate_dataset_normalization(training_dataset,validation_dataset):
    """
    Calculate the dataset normalization parameters, returned as a dict.
    """
    inp_train, gt_train = training_dataset
    inp_val, gt_val = validation_dataset
    
    paired = True if gt_train is not None else False

    # calculate coefficients for weighed average
    c1 = (inp_train.shape[0]) / (inp_train.shape[0] + inp_val.shape[0]) 
    c2 = (inp_val.shape[0]) / (inp_train.shape[0] + inp_val.shape[0]) 

    data_mean = float(np.mean(inp_train) * c1 + np.mean(inp_val) * c2)
    data_std = float(np.std(inp_train) * c1 + np.std(inp_val) * c2)

    if gt_train is not None:
        data_mean_gt = float(np.mean(gt_train) * c1 + np.mean(gt_val) * c2)
        data_std_gt = float(np.std(gt_train) * c1 + np.std(gt_val) * c2)

    model_norm_prm = {
        "data_mean": data_mean,
        "data_std": data_std,
        "data_mean_gt": data_mean_gt if paired else None,
        "data_std_gt": data_std_gt if paired else None
        }  

    return model_norm_prm


def custom_collate_fn(batch):
    """
    Custom collate function that handles tuples of tensors and concatenates them using torch.cat.
    Ensures efficient concatenation with shared memory optimization.

    Args:
        batch (list): List of tuples, where each tuple contains pre-batched 4D tensors (input, ground-truth, time).

    Returns:
        tuple: Concatenated tensors for inputs, ground-truths, and times.
    """
    elem = batch[0]
    if isinstance(elem, tuple):
        transposed = zip(*batch)
        return [custom_collate_fn(samples) for samples in transposed]
    else:
        out = None
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            numel = sum(x.numel() for x in batch)
            storage = elem._typed_storage()._new_shared(numel, device=elem.device)
            out = elem.new(storage).resize_(numel // elem[0].numel(), *elem.size()[1:])
        
        x = torch.cat(batch, dim=0, out=out)
        return x




def load_all_data(data_dir_inp:Path,
                  data_dir_gt:Path=None,
                  data_format:str = "single",
                  val_split = False,
                  make_patches = True,
                  **kwargs):
    """
    Key-worded function that:
        - loads data found in `data_dir_inp`
        - loads data found in `data_dir_gt` if paired dataset
        - adds optional artificial movement 
        - do all the necessary normalization, following `norm_prm`
        - extracts patches from each image (w/ optional patch selection)
        if arg:`make_patches` is True
    
    Returns array of patches ready to be trained:
            (inp_img,gt_img) if paired, (inp_img,None) otherwise

    Arguments:
        - data_dir_inp: Path
            path where to find the input data
        - data_dir_gt: Path (default = None)
            path where to find the gt data for paired dataset       
        - data_format: str (default='single')
            'single','timelapse','mltpl_snr'
        - kwargs should contain all specific data_format info, and normalization parameters
     """
    
    if data_format not in ["single","timelapse","mltpl_snr"]:
        raise ValueError(f"data_format {data_format} unknown.")
    
    if data_dir_gt is not None:
        paired = True
    else:
        paired = False
        
    filters = kwargs.get("filters",None)
    initial_crop=kwargs.get("initial_crop",False)
    mltpl_noise = kwargs.get("mltpl_noise",False)
    select_on_gt = kwargs.get("select_on_gt",False)
    norm_prm = kwargs.get("norm_prm")
    clip = norm_prm.get("clip",False)
    normSig2Obs=norm_prm.get("normSig2Obs",False)
    normalize_data = norm_prm.get("normalize_data",False)
    
    if make_patches:
        patch_thresh = kwargs.get("patch_thresh",None)
        if val_split and kwargs.get("val_patch_size") is not None:
            patch_size = kwargs.get("val_patch_size")
        else:
            patch_size = kwargs.get("patch_size")

    # define normalization parameters
    if isinstance(clip,bool) and clip is True:
        clip = 0

    if not paired and normSig2Obs:
        normSig2Obs = False
        warnings.warn("`normSig2Obs` is True but unpaired dataset")
    if normalize_data or normSig2Obs:
        avgObs = norm_prm.get("avgObs")
        stdObs = norm_prm.get("stdObs")
        if paired:
            avgSig = norm_prm.get("avgSig")
            stdSig = norm_prm.get("stdSig")
    if normSig2Obs:
        if mltpl_noise:
            avgObs_per_noise = kwargs.get("avgObs_per_noise")
            stdObs_per_noise= kwargs.get("stdObs_per_noise")
        else:
            avgObs_per_noise = [avgObs]
            stdObs_per_noise = [stdObs]
    
    # get list of file path
    inp_files = []
    for filter in filters:
        inp_files += sorted(glob.glob(str(data_dir_inp) + f"/*{filter}"))
    if paired:
        gt_files = []
        for filter in filters:
            gt_files += sorted(glob.glob(str(data_dir_gt) + f"/*{filter}"))
        assert len(inp_files) == len(gt_files), f"Found #{len(inp_files)} inp_files and #{len(gt_files)} gt_files"
    
    # loop over all files
    inp_data = []
    gt_data = [] if paired else None
    
    if make_patches:
        n_patch = 0
        if patch_thresh is not None:
            n_patch_removed = 0
        
    for i in range (len(inp_files)):
        inp_file = inp_files[i]
        gt_file = gt_files[i] if paired else None
        
        inp_img,gt_img = load_image(inp_file,gt_file,data_format,**kwargs)
        if inp_img is None:
            continue
        inp_img,gt_img = make_pair_4d (inp_img,gt_img) 

        # artificial movement
        if kwargs.get("artificial_movement",None) is not None:
            prm = kwargs.get("artificial_movement")
            inp_img,gt_img = apply_movement((inp_img,gt_img),prm,volumetric=kwargs.get("volumetric",False))
        
        # clip neg (must be done before sig2obs normalization)
        if not isinstance(clip,bool):
            inp_img[inp_img<clip] = clip
            if paired: gt_img[gt_img<clip] = clip

        # Sig2Obs normalization (for mltpl snr only)
        # if data_format == "mltpl_snr" and paired and normSig2Obs:
        #     for i in range(gt_img.shape[0]):
        #         normalized_frame = (gt_img[i] - avgSig) / stdSig
        #         gt_img[i] = normalized_frame * stdObs_per_noise[i] + avgObs_per_noise[i]

        # when inp already downsampled
        if paired and gt_img.shape[-2:] != inp_img.shape[-2:]: 
            assert gt_img.shape[-1] % inp_img.shape[-1] == 0
            assert gt_img.shape[-1] // inp_img.shape[-1] == \
                        gt_img.shape[-2] // inp_img.shape[-2]
            downsamp_factor = gt_img.shape[-1] // inp_img.shape[-1]
        else:
            downsamp_factor = 1

        # initial crop
        if initial_crop:
            if isinstance(initial_crop,int):
                crop_size = initial_crop//downsamp_factor
            else:
                crop_size = (initial_crop[0]//downsamp_factor,initial_crop[1]//downsamp_factor)
            inp_img = crop_center(inp_img,crop_size)
            if paired: gt_img = crop_center(gt_img,initial_crop)

        # opt. patch extraction and selection -> [patches,SNR,Time,patchsize,patchsize]
        if make_patches: 
            _inp = extract_patches(inp_img,patch_size//downsamp_factor)
            _gt = extract_patches(gt_img,patch_size) if paired else None
            n_patch += _inp.shape[0] *_inp.shape[1]
            
            if patch_thresh is not None:
                for snr in range(_inp.shape[1]): #selection done on each snr independently 
                    _inp_snr = _inp[:,snr,...]
                    _gt_snr = _gt[:,snr,...] if paired else None
                    _inp_selected,_gt_selected,_n_removed = select_patches(_inp_snr,_gt_snr,patch_thresh,
                                                                        select_on_gt=select_on_gt)
                    n_patch_removed+=_n_removed
                    inp_data.append(_inp_selected)
                    if paired: gt_data.append(_gt_selected)
            else:
                inp_data.append(np.concatenate(_inp,axis=0))
                if paired: gt_data.append(np.concatenate(_gt,axis=0))

        else:
            inp_data.append(inp_img)
            if paired: gt_data.append(gt_img)
    
    # patch selection logger info
    if make_patches and patch_thresh is not None:
        logger.info(f"{n_patch_removed}/{n_patch} removed patches, with threshold={patch_thresh}.")

    # transform list(s) into numpy array of all patches¨
    inp_data = np.concatenate(inp_data,axis=0)
    if paired: 
        gt_data = np.concatenate(gt_data,axis=0)

    # full data normalization (optional)
    if normalize_data:
        inp_data = (inp_data - avgObs)/stdObs
        if paired:
            gt_data = (gt_data - avgSig)/stdSig

    return inp_data,gt_data



def load_image(inp_file: Path,
               gt_file: Path=None,
               data_format: str ="single",
               **kwargs):
    """
    Loads pair of images (inp,gt) - gt=None if not paired dataset.
    Images can be 2d, 3d or 4d, depending on the data_format:
        - 2d: single, or mltpl snr with 1 snr
        - 3d: timelapses
        - 4d: mltpl noise levels ([snr,1,h,w])
    
    Arguments:
        - inp_file: Path
        - gt_file: Path (default = None)
        - data_format: str (default = "single")
        - kwargs should contain specific info of the data_format

    Returns:
        - inp_img: np.array
        - gt_img: np.array or None
    
    """
    paired = True if gt_file is not None else False

    inp_img = imread(inp_file)
    gt_img = imread(gt_file) if paired else None 

    if data_format == "single":
        return inp_img,gt_img
        
    elif data_format == "timelapse":
        assert len(inp_img.shape) == 3

        prm = kwargs.get("timelapse_prm",None)
        if prm is None:
            return inp_img,gt_img
        
        if prm.get("timelapse_max_frames",None) is not None:
            assert not paired, "timelapse max frames not implemented for paired dataset"
            nFrames = prm.get("timelapse_max_frames")
            if inp_img.shape[0] > nFrames:
                if prm.get("shuffle",False):
                    idx = np.arange(inp_img.shape[0])
                    np.random.shuffle(idx)
                    inp_img = inp_img[idx]
                inp_img = inp_img[:nFrames]
            
            if prm.get("context_length",None) is None:
                inp_img = np.expand_dims(inp_img,axis=1) # [time,1,h,w] => considered as [snr,1,h,w]
                return inp_img,None

        if prm.get("context_length",None) is not None:
            context_length = prm.get("context_length")
            if context_length == 1:
                return inp_img,gt_img

            if inp_img.shape[0] < context_length:
                name_file = Path(inp_file).name
                print(f"Skipping {name_file} because #frames ({inp_img.shape[0]})<context_length ({context_length})")
                return None,None
            
            side_frames = int((context_length-1)/2)
            inp_imgs = []
            gt_imgs = [] if paired else None
            for idx in range(side_frames,inp_img.shape[0]-side_frames):
                start = idx - side_frames
                stop = idx + side_frames + 1
                inp_imgs.append(inp_img[start:stop])
                if paired:
                    if kwargs.get("volumetric",False):
                        gt_imgs.append(gt_imgs[start:stop]) #volumetric target
                    else:
                        gt_imgs.append(gt_imgs[idx:idx+1]) # single frame target
            inp_imgs = np.stack(inp_imgs,axis=0)
            if paired:
                gt_imgs = np.stack(gt_imgs,axis=0)
            return inp_imgs, gt_imgs
    
    elif data_format == "mltpl_snr":
        prm = kwargs.get("mltpl_snr_prm",None)
        if prm is None or prm.get("snr_idx") is None:
            warnings.warn("`snr_idx` parameter not found.")
            return inp_img,gt_img
        
        snr = prm.get("snr_idx")
        mltplnoise = False
        if isinstance(snr,int):
            assert len(np.shape(inp_img)) == 3 and np.shape(inp_img)[0] > 1
            assert snr < np.shape(inp_img)[0]
        elif isinstance(snr,list):
            assert len(np.shape(inp_img)) == 3 and np.shape(inp_img)[0] >= len(snr)
            mltplnoise=True
        elif snr == "last":
            snr = np.shape(inp_img)[0]-1 #NOTE: could be -1, but then not sure it can be used for sampling_strat_position
        elif snr == "random":
            snr = np.random.randint(low=0,high=np.shape(inp_img)[0])
        else:
            raise ValueError ("Frame idx should be None, an integer,'last',or 'random'")
    
        inp_img = inp_img[snr]
        if paired and len(gt_img.shape) == 3:
            gt_img = gt_img[snr]
        
        if mltplnoise:
            if paired:
                if len(gt_img.shape) == 2:
                    gt_img = np.expand_dims(gt_img,axis=0)
                if gt_img.shape[0] == 1:
                    gt_img = np.repeat(gt_img,repeats=inp_img.shape[0],axis=0)
                else:
                    assert gt_img.shape[0] == inp_img.shape[0], "number of gt and inp frames don't correspond"
            #make 4d [snr,1,h,w]
            inp_img = np.expand_dims(inp_img,axis=1)
            if paired:
                gt_img = np.expand_dims(gt_img,axis=1)

        return inp_img, gt_img


def apply_additional_transforms(list_datasets,inp_transform=None,gt_transform=None):
    
    for i,(inp,gt) in enumerate (list_datasets):
        if inp_transform is not None:
            inp = simple_transforms(inp,inp_transform)
        if gt is not None and gt_transform is not None:
            gt = simple_transforms(gt,gt_transform)
        
        list_datasets[i] = (inp,gt)
    return list_datasets

