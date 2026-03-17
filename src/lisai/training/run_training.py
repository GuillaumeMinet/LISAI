"""Top-level training orchestration entrypoint.

This module wires together the clean training boundaries introduced by the
refactor: resolve the experiment config, initialize the runtime, prepare data,
build the model, construct the trainer, and execute the training loop.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from lisai.config import resolve_config
from lisai.evaluation import run_evaluate

from . import setup
from .runtime import initialize_runtime
from .trainers import get_trainer

if TYPE_CHECKING:
    from lisai.training.trainers.base import TrainingOutcome


POST_TRAINING_INFERENCE_CONFIG = "post_training"


def _prompt_yes_no(prompt: str) -> bool:
    try:
        answer = input(prompt).strip().lower()
    except EOFError:
        return False
    return answer in {"y", "yes"}


def _normalize_training_outcome(outcome) -> "TrainingOutcome":
    from lisai.training.trainers.base import TrainingOutcome

    if outcome is None:
        return TrainingOutcome(reason="completed", last_completed_epoch=None)
    return outcome


def _should_run_post_training_evaluation(cfg, runtime, outcome: "TrainingOutcome") -> bool:
    experiment = getattr(cfg, "experiment", None)
    if not bool(getattr(experiment, "post_training_inference", False)):
        return False
    if runtime.run_dir is None:
        return False
    if outcome.reason in {"completed", "early_stopped"}:
        return True
    if outcome.reason != "interrupted":
        return False
    if outcome.last_completed_epoch is None:
        if runtime.logger is not None:
            runtime.logger.info("Skipping post-training evaluation because no completed epoch is available.")
        return False
    return _prompt_yes_no(
        f"Training interrupted. Run evaluation now using '{POST_TRAINING_INFERENCE_CONFIG}'? [y/N]: "
    )


def _run_post_training_evaluation(cfg, runtime) -> None:
    if runtime.run_dir is None:
        return

    run_evaluate(
        dataset_name=cfg.data.dataset_name,
        model_name=runtime.run_dir.name,
        model_subfolder=cfg.routing.models_subfolder,
        config=POST_TRAINING_INFERENCE_CONFIG,
    )


def run_training(config_path):
    """Run training end to end from a config path and return the trainer instance."""
    cfg = resolve_config(config_path)
    runtime = initialize_runtime(cfg)
    is_volumetric = cfg.model.architecture == "unet3d"

    prepared_data = setup.prepare_data(cfg, runtime)
    setup.save_training_config(
        cfg,
        runtime,
        prepared_data.data_norm_prm,
        prepared_data.model_norm_prm,
    )

    model, state_dict = setup.build_model(
        cfg,
        runtime.device,
        runtime.paths,
        prepared_data.model_norm_prm,
    )

    trainer = get_trainer(
        architecture=cfg.model.architecture,
        model=model,
        train_loader=prepared_data.train_loader,
        val_loader=prepared_data.val_loader,
        device=runtime.device,
        cfg=cfg,
        run_dir=runtime.run_dir,
        volumetric=is_volumetric,
        writer=runtime.writer,
        state_dict=state_dict,
        callbacks=runtime.callbacks,
        patch_info=prepared_data.patch_info,
        console_filter=runtime.console_filter,
        file_filter=runtime.file_filter,
    )

    try:
        outcome = _normalize_training_outcome(trainer.train())
    except Exception:
        runtime.logger.error("Training crashed", exc_info=True)
        raise
    finally:
        if runtime.writer:
            runtime.writer.close()

    try:
        if _should_run_post_training_evaluation(cfg, runtime, outcome):
            _run_post_training_evaluation(cfg, runtime)
    except Exception:
        runtime.logger.error("Post-training evaluation failed", exc_info=True)
        raise

    return trainer
