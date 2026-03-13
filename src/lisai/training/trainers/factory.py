from .lvae import LVAETrainer
from .standard import StandardTrainer


def get_trainer(*, architecture: str, **kwargs):
    if architecture == "lvae":
        return LVAETrainer(**kwargs)
    return StandardTrainer(**kwargs)
