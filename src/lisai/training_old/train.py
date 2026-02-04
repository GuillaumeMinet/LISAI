import pydoc
import torch
import warnings
import logging 
from lisai.lib.utils.logger_utils import CustomStreamHandler
from lisai.lib.utils import get_model, get_paths
from lisai.data.data_prep.make_loaders import make_training_loaders
from lisai.training.helpers import misc,trainer

### CHOOSE CONFIG FILE HERE ###
cfg_file = "cfg_HDN"
###############################

# import cfg file
CFG = pydoc.locate("src.training.configs." + cfg_file + ".CFG") 
if CFG is None:
    raise FileNotFoundError(f"config file {cfg_file} not found.")

# gpu or cpu
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print("Device: ", device)

# set up logger
logging.basicConfig(level="INFO",format='%(name)s: %(levelname)s : %(message)s',handlers=[])
main_log = logging.getLogger('main')
main_log.addHandler(CustomStreamHandler())


local = CFG.get("local")
mode = CFG.get("mode", "train")
exp_name = CFG.get("exp_name","default_exp_name")

# load original CFG if mode is 'continue_training'
if mode=="continue_training":
    exceptions = ["exp_name","training_prm","load_model"] #prm we keep from CFG
    CFG = misc.load_origin_config(CFG,exceptions)

# load noise model for LVAE specific
is_lvae = CFG.get("is_lvae",False)
if is_lvae:
    noiseModel, noiseModel_norm_prm = get_model.getNoiseModel(local,device,CFG.get("noise_model"))
    if CFG.get("normalization").get("load_from_noise_model",False):
        if CFG.get("norm_prm",None) is not None:
            warnings.warn("specified `norm_prm` are not used",
                          "because `load_from_noise_model` is True.")
        CFG["normalization"]["norm_prm"] = noiseModel_norm_prm
else:
    noiseModel = None

#define volumetric parameter
if CFG.get("model_architecture") == "unet3d":
    CFG["data_prm"]["volumetric"] = True
else:
    CFG["data_prm"]["volumetric"] = False

# create data loaders
data_prm = CFG.get("data_prm")
dataset_name=data_prm.get("dataset_name")
norm_prm = CFG.get("normalization").get("norm_prm")
data_dir = get_paths.get_dataset_path(local = local,**data_prm)
train_loader,val_loader,model_norm_prm,patch_info = make_training_loaders(data_dir=data_dir,
                                                                          norm_prm = norm_prm,
                                                                          **data_prm)
CFG["model_norm_prm"] = model_norm_prm

if patch_info is not None:
    data_prm["patch_info"] = patch_info
    CFG["data_prm"] = data_prm

# instantiate model
model,state_dict = get_model.get_model_for_training(CFG,device,model_norm_prm,noiseModel)
print(type(model))

# saving (saving_prm updated with saving folder)
saving_prm = misc.handle_saving(CFG)
CFG["saving_prm"] = saving_prm

# tensorboard writer
writer = misc.handle_tensorboard(CFG)

# init trainer
training_prm = CFG.get("training_prm")

# create trainer
model_trainer = trainer.Trainer(model,train_loader,val_loader,device,
                                state_dict=state_dict,writer=writer,**CFG)

# main log info
main_log.info(f"Mode: {mode}.")
main_log.info(f"Experiment name: {exp_name}")
main_log.info(f"Data: {str(data_dir.parts[-3:])}")
if saving_prm["saving"]:
    main_log.info(f"Model name: {str(saving_prm.get('model_save_folder').parts[-1])}")

model_trainer.train()
