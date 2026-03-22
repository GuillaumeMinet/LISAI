from __future__ import annotations

from typing import Any

from .schema import TrainingSignature


def _normalize_positive_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    if parsed <= 0:
        return None
    return parsed


def _normalize_patch_size(value):
    if value is None:
        return None
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list):
        normalized = [_normalize_positive_int(item) for item in value]
        if any(item is None for item in normalized):
            return None
        return [int(item) for item in normalized if item is not None]
    return _normalize_positive_int(value)


def _effective_train_batch_size(cfg) -> int:
    training_batch = _normalize_positive_int(getattr(getattr(cfg, "training", None), "batch_size", None))
    if training_batch is not None:
        return training_batch
    data_batch = _normalize_positive_int(getattr(getattr(cfg, "data", None), "batch_size", None))
    if data_batch is not None:
        return data_batch
    return 1


def _effective_train_patch_size(cfg):
    data_cfg = getattr(cfg, "data", None)
    if data_cfg is None:
        return None
    return _normalize_patch_size(getattr(data_cfg, "patch_size", None))


def _effective_val_patch_size(cfg, *, train_patch_size):
    data_cfg = getattr(cfg, "data", None)
    if data_cfg is None:
        return train_patch_size
    resolved = _normalize_patch_size(getattr(data_cfg, "val_patch_size", None))
    return train_patch_size if resolved is None else resolved


def _effective_input_channels(cfg) -> int | None:
    model_cfg = getattr(cfg, "model", None)
    if model_cfg is None:
        return None

    architecture = str(getattr(model_cfg, "architecture", "")).strip().lower()
    params = getattr(model_cfg, "parameters", None)
    if params is None:
        return None

    if architecture in {"unet", "unet3d", "rcan"}:
        return _normalize_positive_int(getattr(params, "in_channels", None))
    if architecture == "unet_rcan":
        unet_params = getattr(params, "UNet_prm", None)
        unet_channels = _normalize_positive_int(getattr(unet_params, "in_channels", None))
        if unet_channels is not None:
            return unet_channels
        rcan_params = getattr(params, "RCAN_prm", None)
        return _normalize_positive_int(getattr(rcan_params, "in_channels", None))
    if architecture == "lvae":
        return _normalize_positive_int(getattr(params, "color_ch", None))

    return None


def _effective_upsampling_factor(cfg) -> int | None:
    model_cfg = getattr(cfg, "model", None)
    if model_cfg is None:
        return None
    params = getattr(model_cfg, "parameters", None)
    if params is None:
        return None

    resolved = getattr(params, "effective_upsampling_factor", None)
    if callable(resolved):
        return _normalize_positive_int(resolved())
    return None


def count_trainable_parameters(model) -> int | None:
    params = getattr(model, "parameters", None)
    if not callable(params):
        return None
    try:
        return int(sum(int(p.numel()) for p in params() if bool(getattr(p, "requires_grad", False))))
    except Exception:
        return None


def build_training_signature_from_resolved_config(
    cfg,
    *,
    trainable_params: int | None = None,
) -> TrainingSignature:
    architecture = str(getattr(getattr(cfg, "model", None), "architecture", "")).strip()
    if not architecture:
        architecture = "unknown"

    train_patch_size = _effective_train_patch_size(cfg)
    val_patch_size = _effective_val_patch_size(cfg, train_patch_size=train_patch_size)

    return TrainingSignature(
        architecture=architecture,
        train_batch_size=_effective_train_batch_size(cfg),
        train_patch_size=train_patch_size,
        val_patch_size=val_patch_size,
        input_channels=_effective_input_channels(cfg),
        upsampling_factor=_effective_upsampling_factor(cfg),
        trainable_params=_normalize_positive_int(trainable_params),
    )


__all__ = [
    "build_training_signature_from_resolved_config",
    "count_trainable_parameters",
]
