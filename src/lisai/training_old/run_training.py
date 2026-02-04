import torch
import logging
from lisai.lib.utils.logger_utils import CustomStreamHandler
from lisai.training.helpers import trainer
from lisai.lib.utils import config_utils

def setup_logger():
    logging.basicConfig(
        level="INFO",
        format='%(name)s: %(levelname)s : %(message)s',
        handlers=[]
    )
    log = logging.getLogger('main')
    log.addHandler(CustomStreamHandler())
    return log

def run_training(cfg_or_path):
    """
    Run training from a config dict or config path.
    """

    if isinstance(cfg_or_path, str):
        from lisai.lib.utils import config_utils
        cfg = config_utils.load_yaml_config(cfg_or_path)
    else:
        cfg = cfg_or_path

    # Logger
    log = setup_logger()

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    # Initialize Trainer
    model_trainer = trainer.Trainer.from_config(cfg, device=device)

    # Log experiment info
    exp_name = cfg.get("experiment", {}).get("exp_name", "default_exp")
    log.info(f"Experiment: {exp_name}")
    log.info(f"Mode: {cfg.get('experiment', {}).get('mode', 'train')}")

    # Start training
    model_trainer.train()
    return model_trainer  # return Trainer instance if caller wants to inspect it
