"""Top-level training orchestration entrypoint.

This module wires together the clean training boundaries introduced by the
refactor: resolve the experiment config, initialize the runtime, prepare data,
build the model, construct the trainer, and execute the training loop.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import torch

from lisai.config import resolve_config, resolve_config_dict
from lisai.evaluation import run_evaluate
from lisai.runs import (
    RunMetadataCallback,
    build_training_signature_from_resolved_config,
    count_trainable_parameters,
    create_run_metadata,
    finalize_run_completed,
    finalize_run_failed,
    finalize_run_stopped,
    group_path_from_model_subfolder,
    update_run_heartbeat,
    update_run_runtime_details,
)

from . import setup
from .runtime import initialize_runtime
from .trainers import get_trainer

if TYPE_CHECKING:
    from lisai.config.models import ResolvedExperiment
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


def _log_runtime_warning(runtime, message: str):
    logger = getattr(runtime, "logger", None)
    if logger is None:
        return

    warning = getattr(logger, "warning", None)
    if callable(warning):
        warning(message)
        return

    info = getattr(logger, "info", None)
    if callable(info):
        info(message)


def _safe_run_metadata_call(runtime, action: str, func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception as exc:
        _log_runtime_warning(runtime, f"Run metadata {action} failed: {type(exc).__name__}: {exc}")
        return None


def _build_training_signature_payload(
    cfg,
    *,
    trainable_params: int | None = None,
):
    return build_training_signature_from_resolved_config(
        cfg,
        trainable_params=trainable_params,
    )


def _update_training_signature(
    cfg,
    runtime,
    monitor_enabled: bool,
    *,
    trainable_params: int | None = None,
):
    if not monitor_enabled or runtime.run_dir is None:
        return
    _safe_run_metadata_call(
        runtime,
        "training signature update",
        update_run_runtime_details,
        runtime.run_dir,
        training_signature=_build_training_signature_payload(
            cfg,
            trainable_params=trainable_params,
        ),
    )


def _update_training_timing(
    runtime,
    monitor_enabled: bool,
    *,
    total_training_time_sec: float,
    training_time_per_epoch_sec: float | None,
):
    if not monitor_enabled or runtime.run_dir is None:
        return
    _safe_run_metadata_call(
        runtime,
        "runtime stats update",
        update_run_runtime_details,
        runtime.run_dir,
        total_training_time_sec=total_training_time_sec,
        training_time_per_epoch_sec=training_time_per_epoch_sec,
    )


def _training_time_per_epoch_sec(
    *,
    total_training_time_sec: float | None,
    outcome: "TrainingOutcome",
) -> float | None:
    if total_training_time_sec is None:
        return None
    if outcome.last_completed_epoch is None:
        return None
    epochs_completed = int(outcome.last_completed_epoch) + 1
    if epochs_completed <= 0:
        return None
    return float(total_training_time_sec / epochs_completed)


def _reset_peak_gpu_memory_stats(runtime, monitor_enabled: bool):
    if not monitor_enabled or runtime.run_dir is None:
        return
    device = getattr(runtime, "device", None)
    if getattr(device, "type", None) != "cuda":
        return
    if not torch.cuda.is_available():
        return

    try:
        device_index = device.index if device.index is not None else torch.cuda.current_device()
        torch.cuda.reset_peak_memory_stats(device_index)
    except Exception as exc:
        _log_runtime_warning(runtime, f"Failed to reset CUDA peak memory stats: {type(exc).__name__}: {exc}")


def _persist_peak_gpu_memory_stats(runtime, monitor_enabled: bool):
    if not monitor_enabled or runtime.run_dir is None:
        return
    device = getattr(runtime, "device", None)
    if getattr(device, "type", None) != "cuda":
        return
    if not torch.cuda.is_available():
        return

    try:
        device_index = device.index if device.index is not None else torch.cuda.current_device()
        peak_bytes = int(torch.cuda.max_memory_allocated(device_index))
    except Exception as exc:
        _log_runtime_warning(runtime, f"Failed to read CUDA peak memory stats: {type(exc).__name__}: {exc}")
        return

    peak_mb = (peak_bytes + (1024 * 1024 - 1)) // (1024 * 1024)
    _safe_run_metadata_call(
        runtime,
        "runtime stats update",
        update_run_runtime_details,
        runtime.run_dir,
        peak_gpu_mem_mb=peak_mb,
    )


def _install_run_monitor(cfg, runtime) -> bool:
    if runtime.run_dir is None:
        return False

    model_subfolder = getattr(cfg.routing, "models_subfolder", "")
    group_path = group_path_from_model_subfolder(model_subfolder)
    preserve_existing = getattr(cfg.experiment, "mode", "train") == "continue_training"

    metadata = _safe_run_metadata_call(
        runtime,
        "initialization",
        create_run_metadata,
        runtime.run_dir,
        dataset=cfg.data.dataset_name,
        model_subfolder=model_subfolder,
        max_epoch=getattr(cfg.training, "n_epochs", None),
        group_path=group_path,
        preserve_existing=preserve_existing,
    )
    if metadata is None:
        return False

    _update_training_signature(cfg, runtime, True)

    runtime.callbacks.append(
        RunMetadataCallback(
            runtime.run_dir,
            max_epoch=getattr(cfg.training, "n_epochs", None),
            logger=runtime.logger,
        )
    )
    return True


def _refresh_run_monitor_heartbeat(runtime, monitor_enabled: bool):
    if not monitor_enabled or runtime.run_dir is None:
        return
    _safe_run_metadata_call(runtime, "heartbeat update", update_run_heartbeat, runtime.run_dir)


def _finalize_run_monitor(runtime, monitor_enabled: bool, outcome: "TrainingOutcome"):
    if not monitor_enabled or runtime.run_dir is None:
        return

    if outcome.reason in {"completed", "early_stopped", "no_epochs"}:
        finalizer = finalize_run_completed
    elif outcome.reason == "interrupted":
        finalizer = finalize_run_stopped
    else:
        return

    _safe_run_metadata_call(runtime, "finalization", finalizer, runtime.run_dir)


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
    return run_training_from_resolved_config(resolve_config(config_path))


def run_training_from_config_dict(config: dict):
    """Run training from an in-memory experiment config dictionary."""
    return run_training_from_resolved_config(resolve_config_dict(config))


def run_training_from_resolved_config(cfg: "ResolvedExperiment"):
    """Run training from a resolved experiment config and return the trainer instance."""
    runtime = initialize_runtime(cfg)
    is_volumetric = cfg.model.architecture == "unet3d"
    monitor_enabled = _install_run_monitor(cfg, runtime)

    trainer = None
    trainable_params: int | None = None
    training_start_perf: float | None = None
    try:
        prepared_data = setup.prepare_data(cfg, runtime)
        _refresh_run_monitor_heartbeat(runtime, monitor_enabled)

        setup.save_training_config(
            cfg,
            runtime,
            prepared_data.data_norm_prm,
            prepared_data.model_norm_prm,
        )
        _refresh_run_monitor_heartbeat(runtime, monitor_enabled)

        model, state_dict = setup.build_model(
            cfg,
            runtime.device,
            runtime.paths,
            prepared_data.model_norm_prm,
        )
        trainable_params = count_trainable_parameters(model)
        _update_training_signature(
            cfg,
            runtime,
            monitor_enabled,
            trainable_params=trainable_params,
        )
        _refresh_run_monitor_heartbeat(runtime, monitor_enabled)

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
        _reset_peak_gpu_memory_stats(runtime, monitor_enabled)

        training_start_perf = time.perf_counter()
        outcome = _normalize_training_outcome(trainer.train())
        total_training_time_sec = max(time.perf_counter() - training_start_perf, 0.0)
        _update_training_timing(
            runtime,
            monitor_enabled,
            total_training_time_sec=total_training_time_sec,
            training_time_per_epoch_sec=_training_time_per_epoch_sec(
                total_training_time_sec=total_training_time_sec,
                outcome=outcome,
            ),
        )
    except KeyboardInterrupt:
        _safe_run_metadata_call(runtime, "stop finalization", finalize_run_stopped, runtime.run_dir)
        raise
    except Exception:
        runtime.logger.error("Training crashed", exc_info=True)
        _safe_run_metadata_call(runtime, "failure finalization", finalize_run_failed, runtime.run_dir)
        raise
    finally:
        _persist_peak_gpu_memory_stats(runtime, monitor_enabled)
        if runtime.writer:
            runtime.writer.close()

    _finalize_run_monitor(runtime, monitor_enabled, outcome)

    try:
        if _should_run_post_training_evaluation(cfg, runtime, outcome):
            _run_post_training_evaluation(cfg, runtime)
    except Exception:
        runtime.logger.error("Post-training evaluation failed", exc_info=True)
        raise

    return trainer
