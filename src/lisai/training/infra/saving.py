import json
import os
import warnings
from shutil import copyfile
from lisai.lib.utils import get_paths
from lisai.lib.utils.misc import create_save_folder


def handle_saving(CFG: dict):
    saving_prm = CFG.get("saving_prm")
    if not saving_prm.get("saving", False):
        warnings.warn("Saving disabled")
        return saving_prm

    mode = CFG["mode"]
    local = CFG["local"]

    if mode in ("continue_training", "retrain"):
        origin_model_folder = get_paths.get_model_folder(
            local=local, **CFG["load_model"]
        )

    if mode == "continue_training":
        saving_prm["model_save_folder"] = origin_model_folder
        return saving_prm

    model_save_folder = get_paths.get_model_folder(
        local=local,
        exp_name=CFG["exp_name"],
        dataset_name=CFG["data_prm"]["dataset_name"],
        **saving_prm,
    )
    model_save_folder = create_save_folder(model_save_folder)

    if mode == "retrain":
        origin_save = model_save_folder / "retrain_origin_model"
        os.makedirs(origin_save)
        copyfile(origin_model_folder / LOSS_FILE_NAME, origin_save / "origin_loss.txt")
        copyfile(origin_model_folder / LOG_FILE_NAME, origin_save / "origin_log.txt")
        copyfile(origin_model_folder / CFG_SAVE_NAME, origin_save / "origin_config.json")

    with open(model_save_folder / CFG_SAVE_NAME, "w") as f:
        json.dump(CFG, f, indent=4)

    saving_prm["model_save_folder"] = model_save_folder
    return saving_prm
