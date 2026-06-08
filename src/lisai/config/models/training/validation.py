from __future__ import annotations

import warnings
from typing import Any

from .tasks import (
    DenoisingCARETaskSection,
    DenoisingHDNTaskSection,
    DenoisingUNetRCANTaskSection,
)


def _timelapse_context_length(data: Any) -> int | None:
    timelapse_prm = getattr(data, "timelapse_prm", None)
    if timelapse_prm is None:
        return None
    context_length = getattr(timelapse_prm, "context_length", None)
    return int(context_length) if context_length is not None else None


def _resolved_data_format(data: Any) -> str | None:
    dataset_info = getattr(data, "dataset_info", None)
    if isinstance(dataset_info, dict) and dataset_info.get("data_format") is not None:
        return str(dataset_info["data_format"])
    value = getattr(data, "data_format", None)
    return str(value) if value is not None else None


def _expected_input_channels(data: Any) -> tuple[int | None, str]:
    downsampling = getattr(data, "downsampling", None)
    context_length = _timelapse_context_length(data)

    if downsampling is not None and downsampling.downsamp_method == "multiple":
        if context_length is not None and context_length > 1:
            raise ValueError(
                "`data.downsampling.multiple_prm` is incompatible with `data.timelapse_prm.context_length > 1`."
            )
        n_ch = downsampling.multiple_input_channels()
        if n_ch < 1:
            raise ValueError(
                "`data.downsampling.multiple_prm.fill_factor` yields zero input channels with the current `downsamp_factor`."
            )
        return n_ch, f"`int(downsamp_factor**2 * fill_factor)` = {n_ch}"

    if context_length is not None:
        return context_length, f"`data.timelapse_prm.context_length` ({context_length})"

    if getattr(data, "timelapse_prm", None) is not None:
        return 1, "`data.timelapse_prm.context_length=null` (single-frame timelapse)"

    if _resolved_data_format(data) == "timelapse":
        return None, "timelapse input without explicit `context_length`"

    return 1, "default single-frame input"


def _experiment_task(cfg: Any) -> Any:
    experiment = getattr(cfg, "experiment", None)
    return getattr(experiment, "task", None)


def _require_no_context_window(data: Any, *, task_name: str) -> None:
    context_length = _timelapse_context_length(data)
    if context_length is not None:
        raise ValueError(
            f"`experiment.task.name='{task_name}'` does not support `data.timelapse_prm.context_length`; "
            "remove it or set it to null."
        )


def _effective_upsampling_factor(params: Any) -> int | None:
    factor = getattr(params, "effective_upsampling_factor", None)
    if callable(factor):
        return int(factor())
    return None


def _validate_denoising_task_consistency(
    cfg: Any,
    *,
    architecture: str,
    params: Any,
    data: Any,
) -> None:
    task = _experiment_task(cfg)

    if isinstance(task, DenoisingHDNTaskSection):
        if architecture != "lvae":
            raise ValueError("`experiment.task.name='denoising_hdn'` requires `model.architecture='lvae'`.")
        if bool(data.paired) != bool(task.supervised):
            raise ValueError(
                "`data.paired` must match `experiment.task.supervised` for `denoising_hdn`."
            )
        return

    if isinstance(task, DenoisingCARETaskSection):
        _validate_supervised_denoising_task(
            task_name="denoising_care",
            expected_architecture="unet",
            architecture=architecture,
            params=params,
            data=data,
        )
        return

    if isinstance(task, DenoisingUNetRCANTaskSection):
        _validate_supervised_denoising_task(
            task_name="denoising_unetrcan",
            expected_architecture="unet_rcan",
            architecture=architecture,
            params=params,
            data=data,
        )


def _validate_supervised_denoising_task(
    *,
    task_name: str,
    expected_architecture: str,
    architecture: str,
    params: Any,
    data: Any,
) -> None:
    if architecture != expected_architecture:
        raise ValueError(
            f"`experiment.task.name='{task_name}'` requires `model.architecture='{expected_architecture}'`."
        )
    if not bool(data.paired):
        raise ValueError(f"`experiment.task.name='{task_name}'` requires `data.paired=true`.")
    _require_no_context_window(data, task_name=task_name)
    if getattr(data, "downsampling", None) is not None:
        raise ValueError(f"`experiment.task.name='{task_name}'` does not support `data.downsampling`.")

    upsampling_factor = _effective_upsampling_factor(params)
    if upsampling_factor not in {None, 1}:
        raise ValueError(
            f"`experiment.task.name='{task_name}'` requires model effective upsampling factor 1, "
            f"got {upsampling_factor}."
        )


def validate_cross_section_consistency(cfg: Any, *, emit_warnings: bool) -> Any:
    model = getattr(cfg, "model", None)
    if model is None:
        return cfg

    data = cfg.data
    architecture = model.architecture
    params = model.parameters
    context_length = _timelapse_context_length(data)

    _validate_denoising_task_consistency(
        cfg,
        architecture=architecture,
        params=params,
        data=data,
    )

    if architecture == "lvae" and context_length not in {None, 1}:
        raise ValueError(
            "`model.architecture='lvae'` does not support timelapse context windows; `data.timelapse_prm.context_length` must be null or 1."
        )

    expected_n_ch, source = _expected_input_channels(data)

    if architecture == "lvae" and expected_n_ch not in {None, 1}:
        raise ValueError(
            f"`model.architecture='lvae'` does not support multi-channel generated inputs; got {source}."
        )

    if architecture in {"unet", "unet3d", "rcan"}:
        if expected_n_ch is not None and params.in_channels != expected_n_ch:
            raise ValueError(
                f"`model.parameters.in_channels` must match {source}; expected {expected_n_ch}, got {params.in_channels}."
            )
        if params.out_channels != 1:
            raise ValueError(
                f"`model.parameters.out_channels` must be 1 for architecture '{architecture}', got {params.out_channels}."
            )

    elif architecture == "unet_rcan":
        if expected_n_ch is not None:
            if params.UNet_prm.in_channels != expected_n_ch:
                raise ValueError(
                    f"`model.parameters.UNet_prm.in_channels` must match {source}; expected {expected_n_ch}, got {params.UNet_prm.in_channels}."
                )
            if params.UNet_prm.out_channels != expected_n_ch:
                raise ValueError(
                    f"`model.parameters.UNet_prm.out_channels` must match {source}; expected {expected_n_ch}, got {params.UNet_prm.out_channels}."
                )
        if params.RCAN_prm.out_channels != 1:
            raise ValueError(
                f"`model.parameters.RCAN_prm.out_channels` must be 1 for architecture 'unet_rcan', got {params.RCAN_prm.out_channels}."
            )

    if data.downsampling is not None:
        data_factor = int(data.downsampling.downsamp_factor)
        model_factor = int(params.effective_upsampling_factor())
        if model_factor != data_factor:
            if not bool(data.paired):
                raise ValueError(
                    f"`data.downsampling.downsamp_factor` ({data_factor}) must match the model effective upsampling factor ({model_factor}) when `data.paired=false`."
                )
            if emit_warnings:
                warnings.warn(
                    f"`data.downsampling.downsamp_factor` ({data_factor}) does not match the model effective upsampling factor ({model_factor}).",
                    stacklevel=3,
                )

    return cfg
