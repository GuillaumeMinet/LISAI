import warnings
from pathlib import Path

import torch

from lisai.data.data_prep import make_test_loader
from lisai.evaluation.context import (
    get_model_folder,
    resolve_data_dir,
    resolve_dataset_info,
    resolve_tiling_size,
    resolve_upsampling_factor,
)
from lisai.evaluation.inference.stack import infer_batch
from lisai.evaluation.io import (
    create_save_folder,
    ensure_save_folder,
    save_metrics_json,
    save_outputs,
)
from lisai.evaluation.metrics import compute as metrics
from lisai.infra.config.schema.experiment import DataSection
from lisai.models.loader import get_model_for_inference


def _default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_evaluate(dataset_name:str, 
             model_name:str,
             model_subfolder:str="",
             best_or_last:str = "best",
             epoch_number:int = None,
             test_loader = None,
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
    device = _default_device()

    model_folder = get_model_folder(dataset_name = dataset_name,
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
    else:
        save_folder = ensure_save_folder(Path(save_folder))

    if save_folder is None:
        raise FileNotFoundError("Model folder not found.")

    model,training_cfg,is_lvae = get_model_for_inference(model_folder,device=device,
                                                     best_or_last=best_or_last,
                                                     epoch_number=epoch_number)
    if is_lvae:
        assert lvae_num_samples is not None, ("for LVAE prediction, number of ",
                                              "samples needs to be specified")

    data_prm = dict(training_cfg.get("data_prm") or {})
    norm_prm = (training_cfg.get("normalization") or {}).get("norm_prm")
    model_norm_prm = training_cfg.get("model_norm_prm")
    if isinstance(model_norm_prm, dict):
        model_norm_prm = dict(model_norm_prm)

    upsamp = resolve_upsampling_factor(training_cfg)
    print(f"Found upsampling factor to be: {upsamp}\n")

    if eval_gt is not None and data_prm.get("paired") is False:
        data_prm["paired"] = True
        data_prm["target"] = eval_gt
        if model_norm_prm is None:
            model_norm_prm = {}
        model_norm_prm["data_mean_gt"] = 0
        model_norm_prm["data_std_gt"] = 1
    
    if crop_size is not None:
        data_prm["initial_crop"] = crop_size

    if data_prm_update is not None:
        data_prm.update(data_prm_update)

    tiling_size = resolve_tiling_size(training_cfg, tiling_size)

    if test_loader is None:
        data_dir = resolve_data_dir(training_cfg, data_prm)
        if data_dir is None:
            raise ValueError(
                "Could not resolve `data_dir` for evaluation. "
                "Provide it through `data_prm_update={'data_dir': '...path...'}`."
            )
        dataset_info = resolve_dataset_info(data_prm.get("dataset_name"))
        prep_cfg = DataSection.model_validate(
            data_prm,
        ).resolved(
            data_dir=Path(data_dir),
            norm_prm=norm_prm,
            dataset_info=dataset_info,
            model_norm_prm=model_norm_prm,
            split=split,
        )
        test_loader = make_test_loader(config=prep_cfg)

    
    for batch_id,(x,y) in enumerate(test_loader):
        
        print(f"Image {batch_id} / {len(test_loader)}")

        if torch.isnan(y).all().item():
            y=None
        
        x = x.to(device)
        # x_ = torch.zeros_like(x)
        # x_[:,2] = x[:,2]
        # x = x_.clone()
        print(f"Input shape: {x.shape}")
        
        outputs = infer_batch(
            model,
            x,
            is_lvae=is_lvae,
            tiling_size=tiling_size,
            num_samples=lvae_num_samples,
            upsamp=upsamp,
            ch_out=ch_out,
        )
        tosave = {
            "inp": x.cpu().detach().numpy(),
            "gt": y.cpu().detach().numpy() if y is not None else None,
            "pred": outputs.get("prediction"),
            "samples": outputs.get("samples"),
        }
        save_outputs(tosave, save_folder, img_name=f"img_{batch_id}")

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
            print("Stopping eval because reached limit_n_imgs")
            break
    
    if metrics_list is not None and results is not None:
        save_metrics_json(save_folder, results)
