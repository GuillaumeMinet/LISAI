from types import SimpleNamespace
from lisai.lib.utils import get_paths
from lisai.data.data_prep.make_loaders import make_training_loaders

def prepare(cfg, local):
    """
    Resolves data paths, handles volumetric logic, creates loaders.
    """
    data_prm = cfg.get("data", {})
    norm_prm = cfg.get("normalization", {}).get("norm_prm")

    # 1. Dynamic Logic: Handle Volumetric Flag
    # If unet3d is selected, force volumetric data loading
    if cfg.get("model", {}).get("architecture") == "unet3d":
        data_prm["volumetric"] = True
    else:
        data_prm["volumetric"] = False
    
    # Update config in place so Trainer sees the correct flag later
    cfg["data"]["volumetric"] = data_prm["volumetric"]

    # 2. Resolve Path
    data_dir = get_paths.get_dataset_path(local=local, **data_prm)

    # 3. Create Loaders
    train_loader, val_loader, model_norm_prm, patch_info = make_training_loaders(
        data_dir=data_dir,
        norm_prm=norm_prm,
        **data_prm
    )

    # Return grouped results
    loaders = SimpleNamespace(train=train_loader, val=val_loader)
    
    meta = SimpleNamespace(
        norm_prm=model_norm_prm,
        patch_info=patch_info
    )
    
    return loaders, meta