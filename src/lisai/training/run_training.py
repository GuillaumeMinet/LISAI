# src/lisai/training/run_training.py

import logging
from lisai.config.resolver import resolve_config
from lisai.config.io import save_yaml
from lisai.training import setup
from lisai.training.trainers import get_trainer

def run_training(config_path):
    # 1. Configuration
    cfg = resolve_config(config_path)
    
    # 2. System Setup (Logger, Device, Saving Paths, Writer)
    ctx = setup.system.initialize(cfg)
    
    # Save resolved config for reproducibility
    if ctx.saving_prm.get("saving"):
        save_yaml(cfg, ctx.save_folder / "config_resolved.yaml")

    # 3. Data Setup (Loaders, Stats)
    loaders, meta_data = setup.data.prepare(cfg, ctx.local)

    # 4. Model Setup (Architecture, Noise Models, State Dict)
    model, state_dict = setup.model.build(cfg, ctx.device, meta_data.norm_prm)

    # 5. Trainer Initialization
    trainer = get_trainer(
        model=model,
        train_loader=loaders.train,
        val_loader=loaders.val,
        device=ctx.device,
        # Config sections
        training_prm=cfg.get("training", {}),
        data_prm=cfg.get("data", {}),
        saving_prm=ctx.saving_prm,
        # Experiment context
        exp_name=ctx.exp_name,
        mode=ctx.mode,
        is_lvae=cfg.get("experiment", {}).get("is_lvae", False),
        # Objects
        writer=ctx.writer,
        state_dict=state_dict
    )

    # 6. Execution
    try:
        trainer.train()
    except Exception as e:
        ctx.logger.error("Training crashed", exc_info=True)
        raise e
    finally:
        if ctx.writer:
            ctx.writer.close()

    return trainer