import shutil
import os,sys
from pathlib import Path
import random
import logging
from typing import Union
import json

sys.path.append(os.getcwd())
import lisai.config_data as config_data
import lisai.config_project as config_project
from lisai.lib.utils.get_paths import get_dataset_path
from lisai.data.preprocess.misc.logfile_helpers import parse_log_file
from lisai.data.preprocess.misc.datasets_json_file import update_dataset_json

logger = logging.getLogger('split')


def reverse_split(folder):

    train_dir = Path(folder) / "train"
    train_val = Path(folder) / "test"
    train_test = Path(folder) / "val"

    if os.path.exists(train_dir):
        list = os.listdir(train_dir)
        for item in list:
            shutil.move(Path(train_dir) / item, Path(folder) / item)
        Path(train_dir).rmdir()

    if os.path.exists(train_val):
        list = os.listdir(train_val)
        for item in list:
            shutil.move(Path(train_val)  / item, Path(folder) / item)
        Path(train_val).rmdir()

    if os.path.exists(train_test):
        list = os.listdir(train_test)
        for item in list:
            shutil.move(Path(train_test)  / item, Path(folder) / item)
        Path(train_test).rmdir()


def split(dataset_name,
          preprocess_folder: Path,
          recon:bool,
          raw:bool,
          split_cfg:dict,
          filters:Union[None,dict] = None,
          local:bool = True
          ):

    """ Execute train/val/test split following split_cfg parameters """
    
    assert recon or raw, "At least one of 'recon' and 'raw' should be True"

    # # define filters => i think not useful anymore                                                                                                                                                 if needed
    # if filters is None:
    #     filters = {"raw": ['tif','tiff'], "recon":['h5','hdf5']}
    #     logger.warning(f"No filters given, taking this as filters: {filters}")
        
    # get dataset parameters
    json_file = Path(os.getcwd()) / config_data.CFG["datasets_json"]
    with open(json_file, 'r') as file:
        dataset_json = json.load(file)

    assert dataset_name in dataset_json, "Dataset should be listed in dataset.json"
    dataset_prm = dataset_json[dataset_name]

    mode = split_cfg["mode"]
    mode_prm = split_cfg["mode_parameters"][mode]
    assert mode in ["random","reuse","manual"]

    logger.info (f"splitting mode: {mode} with parameters: {mode_prm}")

    # get number of files per file from dataset_prm
    if recon or (mode == "reuse" and mode_prm["recon_or_raw"] == "recon"):
        n_recon = dataset_prm["size"]["recon"]["n_files"]   
    if raw or (mode == "reuse" and  mode_prm["recon_or_raw"] == "raw"):
        n_raw = dataset_prm["size"]["raw"]["n_files"]

    
    if recon and raw:
        recon_or_raw = 'both'

        if mode == "random":
            if n_recon == n_raw:
                n_split = n_recon
            elif abs(n_recon - n_raw)/max(n_recon,n_raw) < 0.25 :
                n_split = min(n_raw,n_recon)
                logger.warning ("Not same number of files in recon and raw. But difference"
                                " < 25%, still trying to continue with the split.")
            else:
                logger.critical("Difference in number of raw and recon > 25%, stopping split")
                exit()

    elif recon:
        n_split = n_recon
        recon_or_raw = 'recon'

    elif raw:
        n_split = n_raw
        recon_or_raw = 'raw'

    
    # get file names to be moved, depending on the mode.
    # file_names will be a dict where you can access the list of 
    # names to move file_names["recon"/"raw"]["val"/"test"]

    file_names = get_file_names(preprocess_folder,mode, mode_prm,n_split,
                                structure = dataset_prm["structure"],
                                data_format = dataset_prm["data_format"], 
                                recon_or_raw = recon_or_raw, local = local)
    if recon:
        logger.info (f"Recon: validation files: {file_names['recon'].get('val',None)}")
        logger.info (f"Recon: test files: {file_names['recon'].get('test',None)}")
    
    if raw:
        logger.info (f"Raw: validation files: {file_names['raw'].get('val',None)}")
        logger.info (f"Raw: test files: {file_names['raw'].get('test',None)}")


    # move the files
    if recon:
        walk_idx = get_walk_idx(data_format=dataset_prm["data_format"],recon_or_raw="recon")
        n_val,n_test = move_files(folder = preprocess_folder / "recon",
                                  file_names = file_names["recon"],
                                  structure = dataset_prm["structure"]["recon"],
                                  walk_idx=walk_idx)
        
        logger.info ("Successfuly moved recon files")
        update_log_file(file_names["recon"],mode,mode_prm, preprocess_folder, "recon")
        update_dataset_json(dataset_name = dataset_name,data_type = "recon",
                            split = f"{mode} - val: {n_val}, test: {n_test}")

    if raw:
        walk_idx = get_walk_idx(data_format=dataset_prm["data_format"],recon_or_raw="raw")
        n_val,n_test = move_files(folder = preprocess_folder / "raw",
                                  file_names = file_names["raw"],
                                  structure = dataset_prm["structure"]["raw"],
                                  walk_idx= walk_idx)
        
        logger.info ("Successfuly moved raw files")
        update_log_file(file_names["raw"],mode,mode_prm, preprocess_folder, "raw")
        update_dataset_json(dataset_name = dataset_name,data_type = "raw",
                            split = f"{mode} - val: {n_val}, test: {n_test}")

    
def update_log_file(file_names:dict, mode, mode_prm, preprocess_folder:Path, recon_or_raw:str):
    file_name = config_data.CFG["log_names"][recon_or_raw]
    path = Path(preprocess_folder) / file_name

    with open(path,"a") as f:
        f.write(f"\n\n\n------------ SPLIT ------------ \n\n"
                f"{mode} split with parameters: {mode_prm}.\n"
                f"\nMoved files:\n"
                f"\t- val: {file_names['val']}.\n"
                f"\t- test: {file_names['test']}.\n")

def move_files(folder:Path, file_names:dict, structure:str, walk_idx:int):
    """
     Move files in folder according to data structure and list of file names.
     
     structure is expected to be a string with subfolders: "folder1,folder2"
     or an empty string "" if no subfolder.
     
     walk_idx is 1 if the data is organized in folders, such as data_name/file.h5,
     or 2 if the data is organized in files, such as file_name.tif.
    """

    list_val = file_names.get('val',None)
    list_test = file_names.get('test',None)

    structure = list(structure.split(","))
    for _subfolder in structure:
        _folder = folder / _subfolder
        list_all = next(os.walk(_folder))[walk_idx]

        # getting full and/or correct names
        list_val = [file for file in list_all if any(c_x.split('.')[0] in file for c_x in list_val)]
        list_test = [file for file in list_all if any(c_x.split('.')[0] in file for c_x in list_test)]
        
        # create training directory and move all in it
        dir_train = _folder / "train"
        assert os.path.exists(dir_train) == False
        os.makedirs(dir_train)
        for element in list_all:
            shutil.move(_folder / element, dir_train / element )
        
        # create val folder and move validation files
        if list_val is not None:
            dir_val = _folder / "val"
            assert os.path.exists(dir_val) == False
            os.makedirs(dir_val)
            for element in list_val:
                shutil.move(dir_train / element, dir_val / element)

        # create val folder and move validation files
        if list_test is not None:
            dir_test = _folder / "test"
            assert os.path.exists(dir_test) == False
            os.makedirs(dir_test)
            for element in list_test:
                shutil.move(dir_train / element, dir_test / element)

    n_val = len(list_val)
    n_test = len(list_test)
    return n_val,n_test

def get_file_names(preprocess_folder:Path, mode:str, mode_prm:dict,n_split:int,
                   structure:dict, data_format:str, recon_or_raw:str,local:bool):
    
    """ 

    Get "roughly" the files names of the val / split. 
    "roughly" meaning we only need to get the a list containing "cXX", 
    and the right files will be find just before we do the split.
    
    """

    assert mode in ["random","reuse","manual"]

    # define subfolders as a list: raw, recon or both
    if recon_or_raw == 'both':
        subfolders = ["recon","raw"]
    else:
        subfolders = [recon_or_raw]

    # random selection
    if mode == "random":
        n_val = round(mode_prm["val_frc"]*n_split)
        n_test = round(mode_prm["test_frc"]*n_split)
        rdm_idxs = random.sample(range(n_split), n_val + n_test)
        val_idxs = sorted(rdm_idxs[:n_val])
        test_idxs = sorted(rdm_idxs[n_val::])

        file_names = dict()
        for _subfolder in subfolders:
            _folder = preprocess_folder / _subfolder
            walk_idx = get_walk_idx(data_format,recon_or_raw=_subfolder)
            list_data = search_file_names(_folder,structure[_subfolder],walk_idx)

            file_names[_subfolder] = dict()
            file_names[_subfolder]["val"] = [list_data[i] for i in val_idxs ] 
            file_names[_subfolder]["test"] = [list_data[i] for i in test_idxs ] 

    # reusing existing selection
    elif mode == "reuse":
        if mode_prm["dataset_name"] == "same":
            _folder = preprocess_folder
            assert recon_or_raw != 'both'
        else:
            _folder,structure = get_reuse_info(mode_prm["dataset_name"],local) 
        
        _folder = _folder / mode_prm["subfolder"]
        _structure = structure[mode_prm["subfolder"]]
        get_walk_idx(data_format=data_format,recon_or_raw=mode_prm["subfolder"])

        # initiate file_names with correct number of subfolder keys
        file_names = dict()
        for _subfolder in subfolders:
            file_names[_subfolder] = dict()

        # search file names and load them in file_names
        for _split in ["val","test"]:
            _list_names = search_file_names(_folder,_structure,walk_idx,split=_split)
            for _subfolder in subfolders:
                file_names[_subfolder][_split] = _list_names


   # manual selection    
    elif mode == 'manual':
        if mode_prm["original_name"] is True:
            mode_prm["val"] = get_preprocess_names(preprocess_folder,subfolders[0],mode_prm["val"])
            mode_prm["test"] = get_preprocess_names(preprocess_folder,subfolders[0],mode_prm["test"])

        file_names = dict()
        for _subfolder in subfolders:
            file_names[_subfolder] = dict()
            for _split in ["val","test"]:
                file_names[_subfolder][_split] = mode_prm[_split]     

    
    return file_names


def get_preprocess_names(preprocess_folder:Path,recon_or_raw:str,original_names:list):
    """
    Returns preprocess names for each original name found in the "original_names" list.
    """

    # access log file
    file_name = config_data.CFG["log_names"][recon_or_raw]
    path = Path(preprocess_folder) / file_name
    
    #get mapping original --> new 
    mapping = parse_log_file(path)

    # check suffix in the original names of mapping
    first_file = next(iter(mapping))
    suffix = first_file.split('.')[0]

    # get new file names
    new_file_names = []
    for _name in original_names:
        if _name.split('.')[0] != suffix:
            _name = _name.split('.')[0] + suffix
        _new = mapping.get(_name,None)
        if _new is None:
            logger.critical(f"Original name {_name} not foudn")
            continue
        else:
            new_file_names.append(_name)
    
    return new_file_names
        

def get_reuse_info(mode_prm,local):

    # get the path
    dataset_name = mode_prm["dataset_name"]
    path = get_dataset_path(dataset_name,local)

    # get the structure
    json_file = Path(os.getcwd()) / config_data["datasets_json"]
    with open(json_file, 'r') as file:
        dataset_json = json.load(file)

    structure = dataset_json[dataset_name]["structure"]

    return path, structure


def search_file_names(folder,structure,walk_idx,split = ""):
        """
        Search file names in folder / optional split / optional subfolder*
            *optional subfolder should be described in structure[subfolder]
        """

        structure = list(structure.split(','))
        search_folder = Path(folder) / split / structure[0]
        list_data = sorted(next(os.walk(search_folder))[walk_idx])

        return list_data


def get_walk_idx(data_format:str, recon_or_raw:str):
    # NB: for  now assuming possible structures are only ./file or 
    # ./folder/file, will need update if structures get more complex

    saving_template = config_data.CFG["data_formats"][data_format][recon_or_raw]
    if '/' in saving_template:
        return 1
    else:
        return 2    

   
if __name__ == "__main__":


    split_cfg = {
        "mode": 'manual', # 'random', 'reuse', 'manual'

        "mode_parameters":{
            "manual":{
                "original_name": True,
                "val": ["rec_c16_rec_CAM.tiff"],
                "test":[
                    "rec_c02_rec_CAM.tiff",
                    "rec_c16_rec_CAM.tiff",
                    "rec_c27_rec_CAM.tiff",
                ]            
            }
        }    
    }

    split(dataset_name="Vim_fixed_mltplSNR_30nm",
          preprocess_folder = Path(r"E:\dl_monalisa\Data\Vim_fixed_mltplSNR_30nm\preprocess"),
          recon = True,
          raw = False,
          split_cfg = split_cfg)
