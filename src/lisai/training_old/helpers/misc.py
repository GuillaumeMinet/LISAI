import json
from datetime import datetime
import os,sys,warnings
from shutil import copyfile
import torch
sys.path.append(os.getcwd())
from lisai.lib.utils import get_paths,misc
from lisai.lib.utils.misc import create_save_folder

from lisai.config_project import CFG as config_project

_cfg_save_name = config_project.get("cfg_train_filename")
_loss_file_name = config_project.get("loss_file_name")
_log_file_name = config_project.get("train_log_name")


def load_origin_config(CFG:dict,exceptions: list):
    """ 
    Small utils function that updates CFG with original config,
    exept for "local" and "mode", and any keys in list "exceptions".
    Items in exceptions can be list to access nested parameters,
    e.g. exceptions = ["load_model", ["training_prm","lr"]]
    """


    mode = CFG.get("mode")
    local = CFG.get("local")
    assert mode != "train", "cannot load_origin_config if mode is 'train'."

    # get original config
    origin_model_folder = get_paths.get_model_folder(local = local, **CFG["load_model"])
    with open(origin_model_folder / 'config_train.json','r') as f:
        origin_config = json.load(f)

    # updates in origin_config the parameters to keep from CFG
    origin_config ["mode"] = mode
    origin_config ["local"] = local
    for key in exceptions:
        item = misc.nested_get(CFG,key) 
        if item is not None:
            misc.nested_replace(origin_config,key,item)
        # else:
        #     print(f"{key} not found")

    # replace CFG with origin_config
    CFG.update(origin_config)

    return CFG



def handle_saving(CFG:dict):
    """
    Handles saving according to CFG.get("saving_prm).
        - create canonical saving folder
        - saving folder added to "saving_prm"
        - if "retrain", copy original config file to new folder.
        - returns saving_prm
    """

    saving_prm = CFG.get("saving_prm")
    if saving_prm.get("saving",False) is False:
        warnings.warn("`saving` is set to False, nothing will be saved")
        return saving_prm

    mode = CFG.get("mode")
    local = CFG.get("local")
    if mode == "continue_training" or mode == "retrain":
        origin_model_folder = get_paths.get_model_folder(local = local, **CFG["load_model"])
    
    # if continue_training, just need to return existing model folder
    if mode == "continue_training":
        saving_prm["model_save_folder"] = origin_model_folder
        return saving_prm
    
    saving_prm = CFG.get("saving_prm")

    if saving_prm.get("canonical_save",True):
        model_save_folder = get_paths.get_model_folder(local = local, 
                                            exp_name = CFG.get("exp_name"),
                                            dataset_name = CFG.get("data_prm").get("dataset_name"),
                                            **saving_prm)

        model_save_folder = create_save_folder(model_save_folder)
    
    else:
        model_save_folder = saving_prm.get("model_full_path")
        assert model_save_folder is not None, "save name not specified and canonical save False"
        model_save_folder = get_paths.get_model_folder(model_save_folder)

    if mode == 'retrain': # copy origin model config, log and loss files
        origin_save = model_save_folder / 'retrain_origin_model'
        os.makedirs(origin_save)
        copyfile(origin_model_folder / _loss_file_name, origin_save / 'origin_loss.txt')
        copyfile(origin_model_folder / _log_file_name, origin_save / 'origin_log.txt')
        copyfile(origin_model_folder / _cfg_save_name,origin_save / 'origin_config.json')
    
    # save config
    config_save_path = model_save_folder / _cfg_save_name
    with open(config_save_path, 'w') as cfg_json_file: 
        cfg_json_file.write(json.dumps(CFG,indent=4)) 

    saving_prm["model_save_folder"] = model_save_folder
    return saving_prm




def handle_tensorboard(CFG):
    """
    Creates and return tensorbaord writer if:
        CFG.get("tensorboard").get("saving") = True.
    """

    if CFG.get("tensorboard",{"saving": False}).get("saving",False):
        dt = datetime.now().strftime("%d-%m-%Y_%H-%M-%S_")
        exp_name = CFG.get("exp_name")
        from torch.utils.tensorboard import SummaryWriter
        path_tf_runs = get_paths.get_tensorboard_path (
            dataset_name = CFG.get("dataset_name"), 
            subfolder = CFG.get("tensorboard_prm").get("subfolder","")
            )
        writer=SummaryWriter(log_dir=str(path_tf_runs / str(dt + exp_name)))
        return writer
    else:
        return None
    

def make_optimizer_and_scheduler(model, training_prm):
    optimizer = torch.optim.Adam(model.parameters(),training_prm["lr"])
    
    scheduler_str = training_prm.get("scheduler")
    if scheduler_str is None or scheduler_str is False or scheduler_str=="":
        warnings.warn("Scheduler not implemented, set to None instead.")
        return optimizer, None
    
    if scheduler_str == "linear":
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer)
    
    elif scheduler_str == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                     'min',
                                                      patience=10,
                                                      factor=0.5,
                                                      min_lr=1e-12,
                                                      verbose=False)
    else:
        raise ValueError(f"scheduler {scheduler_str} unknown.")
    
    return optimizer,scheduler


