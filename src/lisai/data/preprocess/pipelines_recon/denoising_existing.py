""""
This pipeline is to update existing preprocessed data by applying a denoising NN.
"""

import os,sys,logging
from pathlib import Path
from typing import Union
from tifffile import imsave, imread

sys.path.append(os.getcwd())
from lisai.lib.upsamp.inp_generators import generate_masked_inp
from lisai.evaluation.apply_model import apply_model
from lisai.data.preprocess.misc.datasets_json_file import update_dataset_json
logger = logging.getLogger('pipeline')


def run(dataset_name: str,
        trgt_dir: Path,
        folder_to_apply: str,
        new_name_folder: str,
        model_info: dict,
        downsamp: int = None,
        mask: list = None,
        filters: list = None,
        subfolder: str = "",
        stack_selection_idx: Union[list,str] = None,
        splits: list = ["train","val","test"],
        crop_size: int = None,
        local = True,
        **kwargs):

    origin_folder = trgt_dir / subfolder / folder_to_apply
    target_folder = trgt_dir / subfolder / new_name_folder

    if os.path.exists(target_folder):
        logger.critical(f"{target_folder}  already exists, stopping execution.")
        exit()
    else:
        os.makedirs(target_folder)

    for split in splits:
        src = origin_folder / split
        trgt = target_folder / split

        if mask is not None or (downsamp is not None and downsamp !=1):
            os.makedirs(trgt)
            list_data = os.listdir(src)
            for file in list_data:
                if filters is not None and file.split('.')[-1] not in filters:
                    logger.warning(f'skipping {file} because not in {filters}.')
                    continue 
                im = imread(src / file)
                if len(im.shape) > 2 and stack_selection_idx is not None:
                    im = im[stack_selection_idx]
                if downsamp is not None:
                    im = im[...,::downsamp,::downsamp]
                elif mask is not None:
                    im = generate_masked_inp(im,masking_prm={"mask": mask})[0,0] # 4d array is automatically returned
                save_path = trgt / file
                imsave(save_path,im)
            in_place = True
            data_path = trgt
        else:
            in_place = False
            data_path = src

        apply_model(local=local,
                    model_dataset=model_info.get("model_dataset"),
                    model_subfolder=model_info.get("model_subfolder"),
                    model_name=model_info.get("model_name"),
                    data_path=data_path,
                    stack_selection_idx = stack_selection_idx,
                    lvae_save_samples=False,
                    lvae_num_samples=model_info.get("lvae_num_samples"),
                    in_place=in_place,
                    save_folder=trgt,
                    crop_size = crop_size,
                    tiling_size=200)
    
    #update dataset
    structure = str(target_folder.name)
    update_dataset_json(dataset_name,data_type="recon",structure=structure)