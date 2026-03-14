from lisai.infra.config import resolve_config

from . import setup
from .trainers import get_trainer


def run_training(config_path):
    cfg = resolve_config(config_path)
    ctx = setup.initialize(cfg)
    
    noise_model = None
    if ctx.spec.model_architecture == "lvae":
        noise_model, data_norm_prm = setup.prepare_noise_model(cfg,ctx.device,ctx.paths)
        if data_norm_prm is not None:
            cfg.normalization["norm_prm"] = data_norm_prm

    loaders, meta_data = setup.prepare_data(cfg, ctx)
    model, state_dict = setup.build_model(ctx.spec, ctx.device, meta_data.model_norm_prm, noise_model)
    patch_info = getattr(meta_data, "patch_info", None)

    trainer = get_trainer(
        architecture=cfg.model.architecture,
        model=model,
        train_loader=loaders.train,
        val_loader=loaders.val,
        device=ctx.device,
        cfg = cfg,
        run_dir=ctx.run_dir,
        volumetric=ctx.volumetric,
        writer=ctx.writer,
        state_dict=state_dict,
        callbacks=ctx.callbacks,
        patch_info=patch_info,

        # optional logging filters
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

