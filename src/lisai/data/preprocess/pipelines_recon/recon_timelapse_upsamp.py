import os,sys,logging
from pathlib import Path
from typing import Union, Tuple
from tifffile import imsave, imread
from scipy.ndimage import gaussian_filter

sys.path.append(os.getcwd())
from lisai.data.preprocess.misc.datasets_json_file import update_dataset_json
from lisai.lib.utils.data_utils import bleach_correct_simple_ratio as bc
from lisai.lib.utils.get_paths import get_data_save_path
from lisai.lib.utils.data_utils import crop_center

logger = logging.getLogger('pipeline')

#log file header 
head = str('original_file_name new_file_name').split()
head = f'{head[0]:<30} {head[1]:<30}'

def run(dataset_name: str,
        data_format: str,
        dump_dir: Path,
        trgt_dir: Path,
        log_file_path: Path,
        combine: bool = False,
        subfolder: Path = "",
        crop_size: Union[None,int, Tuple[int, ...]] = None,
        clip_neg: bool = False,
        gauss_filter: Union[None,Tuple] = None,
        remove_first_frame: bool = False,
        bleach_correction: bool = False,
        filters: Union[None,list] = None,
        ):

    assert data_format == "timelapse", "This pipeline should be used only on timelapses"
    
    recon_dir = dump_dir / subfolder
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
        
    count_files = 0
    count_frames = 0
    for folder in list_folders:
        if combine:
            logger.info(f"folder {folder}")
            with open(log_file_path,'a') as f:
                f.write(f"\n\nFolder: {folder}\n")
                f.write(head)

        src_dir = recon_dir / folder

        data_list = sorted(os.listdir(src_dir))

        for i in range(len(data_list)):
            file_name = data_list [i]
            if file_name.split('.')[-1] not in filters:
                logger.warning(f'skipping {file_name} because not in {filters}.')
                continue 

            stack = imread(src_dir / file_name)

            if remove_first_frame:
                stack = stack[1::]
            if bleach_correction:
                stack = bc(stack)
            if crop_size is not None:
                stack = crop_center(stack,crop_size)
            if clip_neg:
                stack[stack<0] = 0
            if gauss_filter is not None:
                stack = gaussian_filter (stack,sigma = gauss_filter[0],radius = gauss_filter[1])
            
            t = stack.shape[0] #number of time points
            new_name = f"c{count_files:02d}"
            save_path = get_data_save_path (trgt_dir,data_format,"recon",new_name,number_of_timepoints=t)
            imsave(save_path, stack)

            count_files += 1
            count_frames += t
            
            logger.info(f"Processed {file_name}.")
            with open(log_file_path,"a") as f:
                f.write(f'\n {file_name:<30} {save_path.name:<30}')
        
    with open(log_file_path,"a") as f:
        f.write(f'\n\nNumber of files: {count_files}.\n'
                f'Total number of frames: {count_frames}.')

    structure = ""

    update_dataset_json(dataset_name,data_type="recon",data_format=data_format,
                        structure=structure,n_files=count_files, n_frames=count_frames)
