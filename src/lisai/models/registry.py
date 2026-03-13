import importlib

MODEL_REGISTRY = {
    "unet": ("lisai.models.unet", "UNet_PosEncod"),
    "unet3d": ("lisai.models.unet3d", "UNet_PosEncod"),
    "rcan": ("lisai.models.rcan", "RCAN"),
    "unet_rcan": ("lisai.models.unet_rcan", "UNetRCAN"),
    "lvae": ("lisai.models.lvae", "LadderVAE"),
}

def get_model_class(name: str):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY.keys())}")
    module_path, class_name = MODEL_REGISTRY[name]
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)
