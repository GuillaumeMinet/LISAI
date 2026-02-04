import pydoc
import torch
import warnings
import json
import os,sys
import numpy as np
from pathlib import Path
sys.path.append(os.getcwd())

from lisai.config_project import CFG as config_project
from lisai.lib.hdn.gaussianMixtureNoiseModel import GaussianMixtureNoiseModel
from .get_paths import get_model_path,get_canonical_folder,get_model_name


from lisai.config_project import CFG as config_project
_trainCFG_name = config_project.get("cfg_train_filename")

def getNoiseModel(local,device,noiseModel_name):
    """
    Small utils function to load noise model according to the canonical way 
    of the project, and to get the normalization parameters if they are found 
    in the noise model folder.

    Parameters
    -----------
    local: bool
        running on local disk or server (see config_project.py)
    device: torch.device object
        device to load noise model (cpu or gpu)
    noiseModel_name: string
        the name of the noise model to load
    
    Returns
    -------
    noiseModel: GaussianMixtureNoiseModel object
        GMM noise model
    norm_prm: dict
        normalization parameters used to create the GMM 
        if found in noise model folder, None otherwise.
    """ 
    
    noiseModel_path = get_canonical_folder(local,"noiseModel",noiseModel_name = noiseModel_name)
    noiseModel_params= np.load(noiseModel_path)
    noiseModel = GaussianMixtureNoiseModel(params = noiseModel_params, device = device)

    path = noiseModel_path.parent / "norm_prm.json"
    if os.path.exists(path):
        with open(str(path)) as f:
            norm_prm = json.load(f)
    else:
        norm_prm = None
    
    print (f"Loaded noise GMM: {noiseModel_name}" )

    return noiseModel, norm_prm


def get_model_for_training(CFG: dict,
              device: torch.device,
              model_norm_prm: dict = None,
              noiseModel=None):
    """
    Create and/or loads existing model to be trained, depending on:
        - CFG.get("mode")
        - CFG.get("load_model").get("load_model")
    
    Parameters
    -----------
    CFG: dict
        training configuration dictionnary
    device: torch.device object
        device to load model (cpu or gpu)
    model_norm_prm: dictionnary, default = None
        Mandatory for LVAE models (used to setup normalization parameters)
    noiseModel: GaussianMixtureNoiseModel object, default = None
        Mandatory for LVAE models.
    
    Returns
    -------
    model: torch object
        model object ready to be trained
    state_dict: dict
        state dictionary of the model if existing, None otherwise.
    """ 
    
    mode = CFG.get("mode")
    local = CFG.get("local")

    if mode == 'continue_training' or mode == 'retrain':
        load_model = True
        load_method = CFG.get("load_model").get("load_method")
        origin_model_path = get_model_path(local = local,train_mode = mode, **CFG["load_model"])
    elif mode == "train":
        load_model = False
        origin_model_path = None
    else:
        raise ValueError(f"Mode {mode} unknown. Should be 'train','continue_training',or 'retrain'.")

    ### Case with full model loading: torch.load and return ###
    if load_model and load_method == 'full_model':
        model = torch.load(origin_model_path)
        if mode == 'continue_training':
            warntext=("Continue training but loading without a state_dict! Epochs start from 0, "
                      "and optimizer/scheduler not initialized with previous state!")
            warnings.warn(warntext)
        return model,None
    

    # other cases: instantiation is needed
    model_prm = CFG.get("model_prm")
    model_architecture = CFG.get("model_architecture")
    img_shape=CFG.get("data_prm").get("patch_size")
    if CFG.get("data_prm").get("downsampling") is not None:
        p =  CFG.get("data_prm").get("downsampling").get("downsamp_factor")
        img_shape = img_shape//p
        
    model = init_model(model_architecture,model_prm,device,
                       model_norm_prm=model_norm_prm,
                       img_shape=img_shape,
                       noiseModel=noiseModel)

    # load state_dict into model
    if load_model:
        state_dict = torch.load(origin_model_path)
        model.load_state_dict(state_dict['model_state_dict'])  
    else:
        state_dict = None

    return model,state_dict



def get_model_for_inference(model_folder:Path, device:torch.device,
                            best_or_last:str = "best",epoch_number: int = None,
                            local:bool = True):
    """
    Loads model ready to be used for inference.
    
    Parameters
    -----------
    model_folder: Path object or str
        full model_folder path
    device: torch.device object
        device to load model (cpu or gpu)
    best_or_last: str, default = "best"
        to use the best or last saved model
    epoch_number: int, default = None
        if provided, will be used to load model with "epoch_{epoch_number}" in the name
    local: bool, default = True
        define if model should be accessed locally or via server
    
    Returns
    -------
    model: torch object
        model object ready to be used for inference
    training_cfg: dictionary
        training configuration found in arg: `model_folder`
    is_lvae: bool
        if the loaded model is LVAE or not
    
    """ 

    model_folder = Path(model_folder)

    with open(str(model_folder / _trainCFG_name)) as f:
        training_cfg = json.load(f)

    model_architecture = training_cfg.get("model_architecture")
    is_lvae = True if model_architecture == "lvae" else False
    if model_architecture == "unetrcan":
        training_cfg["model_architecture"] = "unet_rcan"

    if training_cfg.get("saving_prm").get("state_dict",False) is True:
        load_method = "state_dict"
    else:
        load_method = "full_model"

    # model load
    model_file_name = get_model_name(load_method=load_method,best_or_last=best_or_last,epoch_number=epoch_number)
    
    if load_method == "full_model":
        model = torch.load(model_folder / model_file_name)
        print (f"Loaded model {model_folder.name} - {model_file_name} with 'full_model' method. ")
    else:
        model_architecture = training_cfg.get("model_architecture")
        model_prm =  training_cfg.get("model_prm")
        if is_lvae:
            noiseModel,_ = getNoiseModel(local=local,device=device,
                                       noiseModel_name=training_cfg.get("noise_model"))
            img_shape = training_cfg.get("data_prm").get("patch_size")
            if training_cfg.get("data_prm").get("downsampling") is not None:
                p =  training_cfg.get("data_prm").get("downsampling").get("downsamp_factor")
                img_shape = img_shape//p
            model_norm_prm = training_cfg.get("model_norm_prm")
        else:
            noiseModel = None
            model_norm_prm = None
            img_shape = None
        
        try:
            model = init_model(model_architecture,model_prm,device,
                        model_norm_prm=model_norm_prm,
                        img_shape=img_shape,
                        noiseModel=noiseModel)
            
            state_dict = torch.load(model_folder / model_file_name)

            model.load_state_dict(state_dict['model_state_dict'])

            print (f"Loaded model {model_folder.name} - {model_file_name} with 'state_dict' method. ")
       
        except RuntimeError:
            warnings.warn("Couldn't load with state_dict. Trying with full model load...")
            try:
                model_file_name = get_model_name(load_method="full_model",best_or_last=best_or_last)
                model = torch.load(model_folder / model_file_name)
                print (f"Loaded model {model_folder.name} - {model_file_name} with 'full_model' method. ")
            except FileNotFoundError:
                raise FileNotFoundError("Load state dict not working, and full model file not found either.")

    return model,training_cfg,is_lvae




def init_model(architecture,model_prm,device,model_norm_prm=None,
               noiseModel=None,img_shape=None):
    """
    Import and initiate model.
    
    Parameters:
    ----------
    architecture: str
        model architecture, used to load correct class

    model_prm: dict
        dictionary of the model parameters

    model_norm_prm: dictionnary, default = None
        Mandatory for LVAE models (used to setup normalization parameters)

    noiseModel: GaussianMixtureNoiseModel object, default = None
        Mandatory for LVAE models.

    img_shape: int, default = None
        Mandatory for LVAE models.

    
    Returns:
    -------
    model: torch.Module object
        neural network model
    """

    is_lvae = True if architecture == "lvae" else False

    # get import name
    model_import_name = config_project.get("models_import_name").get(architecture)
    if model_import_name is None:
        raise ValueError(f"Model architecture {architecture} unknown")

    # import model class
    model = pydoc.locate("lisai.models." + model_import_name)
    if model is None:
        raise ImportError (f"Importing model {model_import_name} failed.")
    
    # instantiate model
    if is_lvae:
        assert model_norm_prm is not None, "need model_norm_prm to instantiate new LVAE"
        assert img_shape is not None, "need img_shape to instantiate new LVAE"
        assert noiseModel is not None,"need noiseModel to instantiate new LVAE"
        model_prm["img_shape"]=(img_shape,img_shape)
        model_prm["norm_prm"] = model_norm_prm
        if isinstance(model_prm["z_dims"],int):
            model_prm["z_dims"]= [model_prm["z_dims"]]*int(model_prm["num_latents"])
        model = model(device,**model_prm,noiseModel=noiseModel).to(device)
    else:
        model = model(**model_prm).to(device) 

    return model