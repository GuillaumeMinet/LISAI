import logging
from lisai.lib.utils import get_model

def build(cfg, device, model_norm_prm):
    """
    Instantiates the model, handling specific LVAE noise model requirements.
    """
    logger = logging.getLogger("main")
    
    # 1. LVAE Specific: Noise Model Setup
    noise_model = None
    is_lvae = cfg.get("experiment", {}).get("is_lvae", False)
    
    if is_lvae:
        noise_model = _setup_noise_model(cfg, device, logger)

    # 2. Main Model Instantiation
    # We pass the full cfg as your existing utils likely expect it
    model, state_dict = get_model.get_model_for_training(
        cfg, 
        device, 
        model_norm_prm, 
        noise_model
    )
    
    logger.info(f"Model initialized: {type(model).__name__}")
    return model, state_dict

def _setup_noise_model(cfg, device, logger):
    nm_cfg = cfg.get("noise_model", {})
    local = cfg.get("experiment", {}).get("local", True)
    
    # Load Noise Model
    noise_model, nm_norm_prm = get_model.getNoiseModel(local, device, nm_cfg)
    
    # Logic: Should we use the noise model's stats for normalization?
    load_from_nm = cfg.get("normalization", {}).get("load_from_noise_model", False)
    
    if load_from_nm:
        user_norm = cfg.get("normalization", {}).get("norm_prm")
        if user_norm is not None:
            logger.warning("Config `norm_prm` ignored because `load_from_noise_model` is True.")
        
        # Inject noise model stats into the config so the main model uses them
        cfg["normalization"]["norm_prm"] = nm_norm_prm
        
    return noise_model