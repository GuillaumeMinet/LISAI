from .standard import StandardTrainer
from .lvae import LVAETrainer

def get_trainer(model, train_loader, val_loader, device,
                training_prm=None, data_prm=None, saving_prm=None,
                exp_name=None, mode="train", is_lvae=False,
                writer=None, state_dict=None):
    
    # Common arguments
    kwargs = {
        "model": model,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "device": device,
        "training_prm": training_prm,
        "data_prm": data_prm,
        "saving_prm": saving_prm,
        "exp_name": exp_name,
        "mode": mode,
        "writer": writer,
        "state_dict": state_dict
    }

    if is_lvae:
        return LVAETrainer(**kwargs)
    else:
        return StandardTrainer(**kwargs)