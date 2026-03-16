from lisai.config import resolve_config

from . import setup
from .runtime import initialize_runtime
from .trainers import get_trainer


def run_training(config_path):
    cfg = resolve_config(config_path)
    runtime = initialize_runtime(cfg)
    is_lvae = cfg.model.architecture == "lvae"
    is_volumetric = cfg.model.architecture == "unet3d"

    data_norm_prm = None
    if is_lvae:
        data_norm_prm = setup.resolve_noise_model_metadata(cfg, runtime.paths)

    loaders, meta_data = setup.prepare_data(cfg, runtime, data_norm_prm=data_norm_prm)
    setup.save_training_config(cfg, runtime, data_norm_prm, meta_data.model_norm_prm)

    noise_model = None
    if is_lvae:
        noise_model = setup.load_noise_model_object(
            setup.resolve_noise_model_name(cfg),
            runtime.device,
            runtime.paths,
        )

    model, state_dict = setup.build_model(cfg, runtime.device, meta_data.model_norm_prm, noise_model)
    patch_info = getattr(meta_data, "patch_info", None)

    trainer = get_trainer(
        architecture=cfg.model.architecture,
        model=model,
        train_loader=loaders.train,
        val_loader=loaders.val,
        device=runtime.device,
        cfg=cfg,
        run_dir=runtime.run_dir,
        volumetric=is_volumetric,
        writer=runtime.writer,
        state_dict=state_dict,
        callbacks=runtime.callbacks,
        patch_info=patch_info,
        console_filter=runtime.console_filter,
        file_filter=runtime.file_filter,
    )

    try:
        trainer.train()
    except Exception:
        runtime.logger.error("Training crashed", exc_info=True)
        raise
    finally:
        if runtime.writer:
            runtime.writer.close()

    return trainer
