from lisai.infra.config import resolve_config

from . import setup
from .trainers import get_trainer


def run_training(config_path):
    cfg = resolve_config(config_path)
    ctx = setup.initialize(cfg)

    data_norm_prm = None
    if ctx.spec.model_architecture == "lvae":
        data_norm_prm = setup.resolve_noise_model_metadata(cfg, ctx.paths)

    loaders, meta_data = setup.prepare_data(cfg, ctx, data_norm_prm=data_norm_prm)
    setup.save_training_config(cfg, ctx, data_norm_prm, meta_data.model_norm_prm)

    noise_model = None
    if ctx.spec.model_architecture == "lvae":
        noise_model = setup.load_noise_model_object(
            ctx.spec.noise_model_name,
            ctx.device,
            ctx.paths,
        )

    model, state_dict = setup.build_model(ctx.spec, ctx.device, meta_data.model_norm_prm, noise_model)
    patch_info = getattr(meta_data, "patch_info", None)

    trainer = get_trainer(
        architecture=cfg.model.architecture,
        model=model,
        train_loader=loaders.train,
        val_loader=loaders.val,
        device=ctx.device,
        cfg=cfg,
        run_dir=ctx.run_dir,
        volumetric=ctx.volumetric,
        writer=ctx.writer,
        state_dict=state_dict,
        callbacks=ctx.callbacks,
        patch_info=patch_info,
        console_filter=ctx.console_filter,
        file_filter=ctx.file_filter,
    )

    try:
        trainer.train()
    except Exception:
        ctx.logger.error("Training crashed", exc_info=True)
        raise
    finally:
        if ctx.writer:
            ctx.writer.close()

    return trainer