""""

This pipeline is for fixed dataset "timelapses", with either:

- 1 low ON frame followed by a set of higher ON frames bleaching
- just high ON frames bleaching

=> which case is defined by bool parameter "first_low_inp"

=> will create one or two set of inp folders:
- inp_single (opt.)
- inp_mltpl_snr

and gt folders (depending on "gt" parameter):
- gt_snr0, and/or:
- gt_avg

"""


import os,sys,logging
from pathlib import Path
from typing import Union, Tuple
from tifffile import imsave, imread
from scipy.ndimage import gaussian_filter
import numpy as np

sys.path.append(os.getcwd())
from lisai.lib.utils.get_paths import get_data_save_path
from lisai.data.preprocess.misc.datasets_json_file import update_dataset_json
from lisai.data.preprocess.misc.gather_timelapses import gather_timelapses
from lisai.lib.utils.data_utils import crop_center
from lisai.data.preprocess.misc.registration import sift_registration,pystackreg_registration
logger = logging.getLogger('pipeline')

#log file header 
head = str('original_file_name new_file_name').split()
head = f'{head[0]:<50} {head[1]:<50}'

def run(dataset_name: str,
        data_format: str,
        dump_dir: Path,
        trgt_dir: Path,
        log_file_path: Path,
        combine: bool = False,
        subfolder: Path = "",
        gather_frames: bool = False,
        gather_stridxs: Union[dict,None] = None,
        gt_types: Union[list, None]= None,
        first_low_inp: bool = False,
        registration: bool = True,
        crop_size: Union[None,int, Tuple[int, ...]] = None,
        gt_clip_neg: bool = False,
        gt_gauss_filter: Union[None,Tuple] = None,
        filters: Union[None,list] = None,
        gt_avg_nFrames: int = None,
        ):
    
    assert data_format == "mltpl_snr", "This pipeline should be used only on multiple snr dataset"

    recon_dir = dump_dir / subfolder 

    # get list of folders (if combine is False, dummy list)
    if combine:
        list_folders = os.listdir(recon_dir)
        with open(log_file_path,'a') as f:
            f.write('\nCombining dataset from folders: \n')
            for folder in list_folders:
                f.write(f'\t- {str(folder)}\n')    
            f.write('\n')
        
    else:
        list_folders = [""]
        with open(log_file_path,"a") as f:
            f.write(f'\n\n{head}')
    

    # make inp(s) folder
    if first_low_inp:
        inp_types = ["single","mltpl_snr"]
    else:
        inp_types = ["mltpl_snr"]
    for _type in inp_types:
        _folder = "inp_" + _type
        if os.path.exists(trgt_dir / _folder):
                logger.critical(f"{_folder}  already exists, stopping execution."
                            "Set overwrite = True in preprocess parameters to overwrite existing folders.")
                exit()
        else:
            os.makedirs(trgt_dir / _folder)

    # make gt folder(s)
    if gt_types is not None:
        for _type in gt_types:
            assert _type in ["snr0","avg"], "snr0 and avg are the only known types of gt"
            _folder = "gt_" + _type
            if os.path.exists(trgt_dir / _folder):
                    logger.critical(f"{_folder} subfolder already exists, stopping execution."
                                "Set overwrite = True in preprocess parameters to overwrite existing folders.")
                    exit()
            else:
                os.makedirs(trgt_dir / _folder)
    
    else:
        logger.warning("Multiple snr pipeline ongoing without creating a GT!")


    # looping over folders (dummy loop if combine is False)
    n_snr_list = []
    count_files = 0
    for folder in list_folders:
        if combine:
            logger.info(f"folder {folder}")
            with open(log_file_path,'a') as f:
                f.write(f"\n\nFolder: {folder}\n")
                f.write(head)


        src_dir = recon_dir / folder
        
        # gathering frames (potentially updating src_dir if new frames saved in subfolder)
        if gather_frames:
            src_dir = gather_timelapses(src_dir,stack_name_start=gather_stridxs["start"],
                                        stack_name_end=gather_stridxs["end"],overwrite=True)

        # get list of data
        data_list = sorted(os.listdir(src_dir))

        # process each data
        for i in range(len(data_list)):
            file_name = data_list [i]
            if file_name.split('.')[-1] not in filters:
                logger.warning(f'skipping {file_name} because not in {filters}.')
                continue 

            stack = imread(src_dir / file_name)
            assert len(stack.shape) == 3 and stack.shape[0] > 1

            if registration:
                if first_low_inp:
                    ref_frame = 1
                else:
                    ref_frame = 0
                #stack = sift_registration(stack,ref_frame_idx=ref_frame)
                stack = pystackreg_registration(stack) # NB: reference will always be first frame for now

            if crop_size is not None:
                stack = crop_center(stack,crop_size)
            
            # create inp(s)
            inp_dict = dict()
            if "single" in inp_types:
                snr0_idx = 1
                inp_dict["single"] = stack[0]
                inp_dict["mltpl_snr"] = stack[1:]
            else:
                snr0_idx = 0
                inp_dict["mltpl_snr"] = stack
                
            # create gt(s) 
            if gt_types is not None:
                gt_dict = dict()
            if "snr0" in gt_types:
                gt_dict["snr0"] = stack[snr0_idx]
            if "avg" in gt_types:
                gt_dict["avg"] = np.mean(stack[:gt_avg_nFrames],0)

            # apply gt specific preprocesses
            if gt_types is not None:
                for key in gt_dict:
                    if gt_clip_neg:
                        _gt = gt_dict[key]
                        _gt[_gt<0] = 0
                        gt_dict[key] = _gt
                    if gt_gauss_filter is not None and gt_gauss_filter is not False :
                        gt_dict [key] = gaussian_filter (gt_dict [key],
                                                         sigma = gt_gauss_filter[0],
                                                         radius = gt_gauss_filter[1])
            
            new_name = f"c{count_files:02d}"

            # saving inp(s)
            for key in inp_dict:
                save_path = get_data_save_path (trgt_dir / ("inp_" + key), data_format,
                                        "recon",new_name)
                
                imsave(save_path,inp_dict[key])

            # saving gt(s)
            if gt_types is not None:
                for key in gt_dict:
                    save_path_gt = get_data_save_path (trgt_dir / ("gt_" + key), data_format,
                                           "recon",new_name)
                    imsave(save_path_gt,gt_dict[key])
            

            # update count, logger and txt file
            count_files += 1
            logger.info(f"Processed {file_name}.")
            with open(log_file_path,"a") as f:
                f.write(f'\n {file_name:<50} {save_path.name:<50}')

            # update n_snr list
            if stack.shape[0] not in n_snr_list:
                n_snr_list.append(stack.shape[0])
        
        # update dataset
        structure = ",".join(["inp_" + key for key in inp_types] \
            + ["gt_" + key for key in gt_types])

        update_dataset_json(dataset_name,data_type="recon",data_format=data_format,
                        structure=structure,n_files=count_files,n_snr = n_snr_list)

    