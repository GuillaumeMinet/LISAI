import os,sys
from pathlib import Path
import logging

sys.path.append(os.getcwd())
from src import config_data,config_project

logger = logging.getLogger("get_canonical_paths")

_code_dir = os.getcwd()

def get_model_path(**kwargs):
    
    canonical_load = kwargs.get("canonical_load", None)
    if canonical_load is not None and canonical_load is not False:
        model_folder = get_model_folder(**kwargs)
        if kwargs.get("model_name",None) is not None and len(str(kwargs.get("model_name"))) != 0:
            model_name = kwargs.get("model_name")
        else:
            model_name = get_model_name(**kwargs)

        return model_folder / model_name

    else:
        full_path = kwargs.get("model_full_path",None)
        assert full_path is not None and len(str(full_path)) != 0
        return kwargs.get("model_full_path")



def get_model_folder(local:bool,**kwargs):
    """
    Return model_folder path (pathlib.Path object).
    """
    folder = get_canonical_folder(local,
                                  structure_type="model_saving",
                                  **kwargs)

    return folder
   


def get_model_name(**kwargs):
    """
    Gets model name (str)
        
    Args:
        - load_method: str
            'state_dict' or 'full_model'
        
        - best_or_last: str
            'best' or 'last'
            mutually exclusive with arg:`train_mode`
            "model_best..." or "model_last..."

        - train_mode: str
            "continue_training","retrain"
            mutually exclusive with arg:`best_or_last`
                --> if "continue_training" we use "model_last" as prefix
                --> if "retrain" we use "model_best" as prefix
                
        - epoch_number: int
            if provided, will be used as "epoch_{epoch_number}" in the name
            mutually exclusive with arg:`best_or_last` and arg:`train_mode`

    """

    load_method = kwargs.get("load_method",None)
    train_mode = kwargs.get("train_mode",None)
    best_or_last = kwargs.get("best_or_last",None)
    epoch_number = kwargs.get("epoch_number",None)

    assert load_method is not None
    assert train_mode is not None or best_or_last is not None or epoch_number is not None
    
    if train_mode is None:
        if epoch_number is not None:
            middle = f"epoch_{epoch_number}"
        else:
            assert best_or_last in ["best","last"]
            middle = best_or_last
    else:
        if train_mode == "continue_training":
            middle = "last"
        elif train_mode == "retrain":
            middle = "best"
        elif train_mode == "train":
            raise ValueError ("train_mode cannot be 'train' when loading model")
        else:
            raise ValueError ("train_mode unknown")

    if load_method == "state_dict":
        return f"model_{middle}_state_dict.pt"
    elif load_method == "full_model":
        return f"model_{middle}.pt"
    else:
        raise ValueError ("load_method unknown")
    

def get_dataset_path(dataset_name:str=None,
                     canonical_load:bool = True,
                     subfolder:str = None,
                     full_data_path:str = None,
                     local:bool = False,
                     **kwargs,
                     ):
    
    """
    Return dataset_path (`Path` object).
    """
    
    if canonical_load == True:
        assert dataset_name is not None, "Need dataset name for canonical load"
        
        
        if full_data_path is not None and len(str(full_data_path)) != 0:
            logger.warning("Ignoring full data path argument because canonical load is True")

        folder = get_canonical_folder(local=local,
                                      structure_type="dataset_loading",
                                      dataset_name = dataset_name,
                                      subfolder = subfolder)
    
        return folder

    else:
        assert full_data_path is not None
        return Path(full_data_path)


def get_data_save_path(trgt_dir: Path, 
                  data_format: str,
                  data_type: str, 
                  name:str,
                  snrlvl: int= None,
                  number_of_timepoints: int =None
                  ):
    """
    Returns full saving path according to the data_format and data_type, 
    and the canonical way of saving as defined in config_data.CFG.
    If needed, it creates the folder where the data should be saved (e.g. for raw timelapses).
    """
    
    if data_format not in config_data.CFG['data_formats']:
        raise ValueError(f"Unknown data_format: {data_format}")
    
    # Get the format pattern from the configuration
    saving_format = config_data.CFG['data_formats'][data_format][data_type]

    # Replace placeholders with actual values
    saving_format = saving_format.replace('*name*', name)
    if '*snrlvl*' in saving_format and snrlvl is not None:
        saving_format = saving_format.replace('*snrlvl*', str(snrlvl))
    if '*number-of-timepoints*' in saving_format and number_of_timepoints is not None:
        saving_format = saving_format.replace('*number-of-timepoints*', f"{number_of_timepoints:02d}")
    
    directory = Path(os.path.dirname(trgt_dir /  saving_format)) #FIXME: should be a better way to do this
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    return directory / saving_format




def get_tensorboard_path (**kwargs):
    """
    Return tensorboard path (`Path` object).
    """
    folder = get_canonical_folder(local = False,
                                  structure_type="tensorboard_saving",
                                  **kwargs)
    
    return folder



def get_infer_saving_folder(local:bool,**kwargs):
    """
    Return inference saving folder (`Path` object).
    """
    folder = get_canonical_folder(local,structure_type="inference_saving",**kwargs)
    
    #add optional split name
    if kwargs.get("split",None) is not None:
        folder = folder.parent / (folder.name + "_" + kwargs.get("split"))

    # add optional best_or_last
    if kwargs.get("best_or_last",None) is not None and kwargs.get("best_or_last") != "":
        folder = folder.parent / (folder.name + "_" + kwargs.get("best_or_last"))

    # add optional suffix to folder name
    if kwargs.get("suffix", None) is not None and kwargs.get("suffix") != "":
        folder = folder.parent / (folder.name + "_" + kwargs.get("suffix"))
    
    
    return folder



def get_canonical_folder(local:bool,structure_type:str,**kwargs):
    """
    Genereal function returning the canonical folder, 
    as setup in the config_project, in the "structure" dictionary,
    in the arg:`structure_type` key.
    """

    pass_list = ["subfolder"] # list of placeholders skipped if we don't find them
    config = config_project.CFG

    # local or server run
    if local:
        root = config["data_root"]["local"]    
    else:
        root = config["data_root"]["server"]

    # get folder name with place holders
    folder_name =config["structure"].get(structure_type,None)
    if folder_name is None:
        raise ValueError ("Structure type not referenced in config_project.py")
    
    # replace placeholders
    # NOTE: 
    #   - if placeholders in singel quote '..', not replaced.
    #   - if placeholder == "code_dir", it is replace by os.getcwd()
    #   - placeholders searched in the kwargs first, then in the config_project dict.
    #   - placeholders not found will throw an error, except if in "pass_list"

    parts = folder_name.split('/')
    new_parts = []
    for placeholder in parts:
        if placeholder.startswith("'") and placeholder.endswith("'"):
            new_parts.append(placeholder.split("'")[1])
        elif placeholder == "code_dir":
            new_parts.append(str(_code_dir))
        else:
            if kwargs.get(placeholder, None) is not None:
                new_parts.append(kwargs.get(placeholder))
            elif config_project.CFG.get(placeholder, None) is not None:
                new_parts.append(config_project.CFG.get(placeholder))
            else:
                if placeholder not in pass_list:
                    raise ValueError(f"{placeholder} not found")
    
    folder_name = '/'.join(new_parts)

    return root / Path(folder_name)