import torch
import logging
from types import SimpleNamespace
from lisai.lib.utils import misc
from lisai.lib.utils.logger_utils import CustomStreamHandler

def initialize(cfg):
    """
    Sets up Logging, Device, Saving paths, and Tensorboard.
    Returns a context object with these attributes.
    """
    exp_cfg = cfg.get("experiment", {})
    
    # Context container
    ctx = SimpleNamespace()
    ctx.exp_name = exp_cfg.get("exp_name", "default")
    ctx.mode = exp_cfg.get("mode", "train")
    ctx.local = exp_cfg.get("local", True)
    
    # 1. Saving Paths (Resolves paths in 'saving' section)
    ctx.saving_prm = misc.handle_saving(cfg)
    ctx.save_folder = ctx.saving_prm.get("model_save_folder")

    # 2. Logger
    ctx.logger = _setup_logger(ctx.save_folder, ctx.exp_name)
    ctx.logger.info(f"--- Experiment: {ctx.exp_name} | Mode: {ctx.mode} ---")

    # 3. Device
    use_cuda = torch.cuda.is_available()
    ctx.device = torch.device("cuda" if use_cuda else "cpu")
    ctx.logger.info(f"Device: {ctx.device}")

    # 4. Tensorboard
    ctx.writer = misc.handle_tensorboard(cfg)

    return ctx

def _setup_logger(save_folder, exp_name):
    # We configure the "lisai" logger so all child loggers inherit this setting
    logger = logging.getLogger("lisai") 
    logger.setLevel(logging.INFO)
    logger.handlers = [] 

    # 1. Console Handler
    logger.addHandler(CustomStreamHandler())

    # 2. File Handler (Only place this happens!)
    if save_folder:
        log_path = save_folder / "train_log.log"
        file_handler = logging.FileHandler(log_path)
        fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
    
    return logger