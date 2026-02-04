import os,sys
import torch
import warnings
import json
from pathlib import Path

sys.path.append(os.getcwd())
from lisai.lib.utils.misc import create_save_folder
from lisai.lib.utils.get_paths import get_model_folder
from lisai.lib.utils.get_model import get_model_for_inference
from lisai.data.data_prep.make_loaders import make_test_loader
from lisai.evaluation.helpers import metrics,eval_utils,predict
from lisai.config_project import CFG

_default_tiling_size = CFG.get("default_tiling_size")

# gpu or cpu
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print("Device: ", device)

def evaluate(dataset_name:str, 
             model_name:str,
             model_subfolder:str="",
             best_or_last:str = "best",
             epoch_number:int = None,
             test_loader = None,
             local:bool = True, 
             tiling_size:int = None,
             crop_size:int = None,
             metrics_list:list = None,
             lvae_num_samples:int = None,
             results:dict = None,
             save_folder: Path = None,
             overwrite: bool = False,
             eval_gt = None,
             data_prm_update = None,
             ch_out = None,
             split = "test",
             limit_n_imgs = None
             ):

    model_folder = get_model_folder(local=local,
                                    dataset_name = dataset_name,
                                    subfolder = model_subfolder,
                                    exp_name = model_name)
    
    if save_folder is None:
        if epoch_number is not None:
            save_name = f"evaluation_epoch{epoch_number}"
        else:
            save_name = f"evaluation_{best_or_last}"
        if split != "test":
            save_name = f"{save_name}_{split}"
        save_folder = create_save_folder(path = model_folder/ save_name,
                                         overwrite=overwrite,parent_exists_check=True)
    if save_folder is None:
        raise FileNotFoundError("Model folder not found.")

    model,training_cfg,is_lvae = get_model_for_inference(model_folder,device=device,
                                                     best_or_last=best_or_last,
                                                     epoch_number=epoch_number,
                                                     local=local)
    if is_lvae:
        assert lvae_num_samples is not None, ("for LVAE prediction, number of ",
                                              "samples needs to be specified")

    data_prm = training_cfg.get("data_prm")
    norm_prm = training_cfg.get("normalization").get("norm_prm")
    model_norm_prm = training_cfg.get("model_norm_prm")

    possible_keys = ["upsamp", "upsampling_factor"]
    upsamp = next((training_cfg.get("model_prm").get(key) for key in possible_keys if training_cfg.get("model_prm").get(key) is not None), None)
    if upsamp is None:
        upsamp = 1
    print(f"Found upsampling factor to be: {upsamp}\n")

    if eval_gt is not None and data_prm.get("paired") is False:
        data_prm["paired"] = True
        data_prm["gt"] = eval_gt
        model_norm_prm["data_mean_gt"] = 0
        model_norm_prm["data_std_gt"] = 1
    
    if crop_size is not None:
        data_prm["initial_crop"] = crop_size

    if data_prm_update is not None:
        data_prm.update(data_prm_update)

    if tiling_size is None:
        tiling_size = _default_tiling_size.get(training_cfg.get("model_architecture"))

    if test_loader is None:
        test_loader = make_test_loader(norm_prm=norm_prm,
                                       model_norm_prm=model_norm_prm,
                                       split = split,
                                       **data_prm)

    
    for batch_id,(x,y) in enumerate(test_loader):
        
        print(f"Image {batch_id} / {len(test_loader)}")

        if torch.isnan(y).all().item():
            y=None
        
        x = x.to(device)
        # x_ = torch.zeros_like(x)
        # x_[:,2] = x[:,2]
        # x = x_.clone()
        print(f"Input shape: {x.shape}")
        
        outputs = predict.predict(model,x,is_lvae=is_lvae,
                                  tiling_size=tiling_size,
                                  num_samples=lvae_num_samples,
                                  upsamp=upsamp,ch_out=ch_out)
        tosave = {
            "inp": x.cpu().detach().numpy(),
            "gt": y.cpu().detach().numpy() if y is not None else None,
            "pred": outputs.get("prediction"),
            "samples": outputs.get("samples"),
        }
        eval_utils.save_outputs(tosave, save_folder,img_name = f"img_{batch_id}" )

        if metrics_list is not None and y is None:
            warnings.warn("no ground-truth provided, cannot calculate metrics")

        elif metrics_list is not None:
            if x.shape[-2:] == y.shape[-2:]:
                inp = x.cpu().detach().numpy()
            else:
                inp = None

            gt = y.cpu().detach().numpy()
            results = metrics.calculate_metrics(img_name=f"img_{batch_id}",
                                                        metrics=metrics_list,
                                                        results=results,
                                                        pred = outputs.get("prediction"),
                                                        gt = gt,inp = inp)
        if limit_n_imgs is not None and batch_id >= limit_n_imgs-1:
            print(f"Stopping eval because reached limit_n_imgs")
            break
    
    if metrics_list is not None and y is not None:
        with open(save_folder/"metrics.json", 'w') as f:
            json.dump(results, f, indent=4)



if __name__ == "__main__":
    
    list_folders = ["Avg_unpaired_Mltpl075_UnetRCAN_rg8_rcab12_red16_CharEdge_Augm"]

    # list_folders = ["Snr0_unpaired_Mltpl05_UnetRCAN_rg8_rcab12_red16_CharEdge_alpha005_upsampKernel4"]
    
    # list_folders = [
    #     "HDN_single_GMMsigN2VAvgbis_KL03_noAugm",
    #     "HDN_single_GMMsigN2VAvgbis_KL05_noAugm",
    #     "HDN_single_GMMsigN2VAvgbis_KL07_noAugm",
    # ]

    # list_folders = "all"
    # exceptions = None#["SNR0","SNRavg","SNR1"]
    
    # folder = r"E:\dl_monalisa\Models\Vim_fixed_mltplSNR_30nm\Upsampling_selected\unpaired"
    
    # if list_folders == "all":
    #     list_folders = [f for f in os.listdir(folder) if os.path.isdir(os.path.join(folder, f))]

    fails = {}
    for model in list_folders:
        try:
            if model in exceptions:
                continue
        except:
            pass

        print(f"Evaluating model {model}:")
        try:
            evaluate(dataset_name = "Vim_fixed_mltplSNR_30nm", 
                    model_name = model,
                    model_subfolder = r"Upsampling_selected\unpaired",
                    metrics_list = None,
                    tiling_size=300,
                    eval_gt = None,
                    best_or_last = "best",
                    epoch_number=None,
                    crop_size=None,#(334,1238),
                    ch_out=1,
                    overwrite = True,
                    lvae_num_samples=10,
                    limit_n_imgs = None,
                    data_prm_update = None,
                    split="test")
                
        except Exception as e:
            print(e)
            fails[model] = e
        
    for model,error in fails.items():
        print(f"Model {model} got following error during evaluation:\n {error}")