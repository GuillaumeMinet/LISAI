from __future__ import annotations

import warnings
from typing import Any, Dict

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .data import DataSection, ExperimentDataSection
from .loading import ExperimentLoadModelSection, LoadModelSection
from .loss import LossFunctionConfig
from .model import ModelSection
from .normalization import NormalizationSection
from .sections import (
    ExperimentSection,
    NoiseModelSection,
    ResolvedExperimentSection,
    RoutingSection,
    SavingSection,
    TensorboardSection,
    TrainingSection,
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

    if getattr(data, "timelapse_prm", None) is not None or _resolved_data_format(data) == "timelapse":
        return None, "timelapse input with `context_length=null`"

    return 1, "default single-frame input"



def _validate_cross_section_consistency(cfg: Any, *, emit_warnings: bool) -> Any:
    model = getattr(cfg, "model", None)
    if model is None:
        return cfg

    data = cfg.data
    architecture = model.architecture
    params = model.parameters
    context_length = _timelapse_context_length(data)

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


class ExperimentConfig(BaseModel):
    """User-authored training experiment YAML schema."""

    model_config = ConfigDict(extra="allow")

    experiment: ExperimentSection = Field(
        default_factory=ExperimentSection,
        description="High-level experiment metadata and execution mode.",
    )
    routing: RoutingSection = Field(
        default_factory=RoutingSection,
        description="Subfolder routing that determines where data, models, logs, and inference outputs live.",
    )
    data: ExperimentDataSection = Field(
        default_factory=ExperimentDataSection,
        description="User-authored data-loading, patching, and preprocessing settings.",
    )
    model: ModelSection | None = Field(
        default=None,
        description="Model architecture selection and typed constructor parameters.",
    )
    training: TrainingSection = Field(
        default_factory=TrainingSection,
        description="Core optimization settings for the training loop.",
    )
    normalization: NormalizationSection = Field(
        default_factory=NormalizationSection,
        description="Dataset normalization settings applied before or during data loading.",
    )
    model_norm_prm: Dict[str, Any] | None = Field(
        default=None,
        description="Optional model-output normalization settings kept separate from the main data normalization block.",
    )
    loss_function: LossFunctionConfig | None = Field(
        default=None,
        description="Typed loss-function configuration used to instantiate the trainer loss.",
    )
    noise_model: NoiseModelSection | str | None = Field(
        default=None,
        description="Optional noise-model configuration or shortcut name.",
    )
    saving: SavingSection = Field(
        default_factory=SavingSection,
        description="Checkpoint and validation-image saving behavior.",
    )
    tensorboard: TensorboardSection = Field(
        default_factory=TensorboardSection,
        description="TensorBoard logging settings.",
    )
    load_model: ExperimentLoadModelSection = Field(
        default_factory=ExperimentLoadModelSection,
        description="Optional source-run selection used to initialize the model from a previous checkpoint.",
    )

    @model_validator(mode="after")
    def _validate_cross_section_rules(self):
        return _validate_cross_section_consistency(self, emit_warnings=False)


class ResolvedExperiment(BaseModel):
    """Runtime training config after merge and normalization."""

    model_config = ConfigDict(extra="allow")

    experiment: ResolvedExperimentSection = Field(
        default_factory=ResolvedExperimentSection,
        description="Resolved experiment metadata enriched with runtime-only bookkeeping.",
    )
    routing: RoutingSection = Field(
        default_factory=RoutingSection,
        description="Resolved routing configuration for datasets, models, logs, and inference outputs.",
    )
    data: DataSection = Field(
        default_factory=DataSection,
        description="Resolved data-loading and preprocessing settings used by the runtime.",
    )
    model: ModelSection | None = Field(
        default=None,
        description="Resolved model architecture selection and typed constructor parameters.",
    )
    training: TrainingSection = Field(
        default_factory=TrainingSection,
        description="Resolved optimization settings for the training loop.",
    )
    normalization: NormalizationSection = Field(
        default_factory=NormalizationSection,
        description="Resolved dataset normalization settings used during training data preparation.",
    )
    model_norm_prm: Dict[str, Any] | None = Field(
        default=None,
        description="Resolved model-output normalization settings.",
    )
    loss_function: LossFunctionConfig | None = Field(
        default=None,
        description="Resolved typed loss-function configuration.",
    )
    noise_model: NoiseModelSection | str | None = Field(
        default=None,
        description="Resolved optional noise-model configuration or shortcut name.",
    )
    saving: SavingSection = Field(
        default_factory=SavingSection,
        description="Resolved checkpoint and artifact-saving behavior.",
    )
    tensorboard: TensorboardSection = Field(
        default_factory=TensorboardSection,
        description="Resolved TensorBoard logging settings.",
    )
    load_model: LoadModelSection = Field(
        default_factory=LoadModelSection,
        description="Resolved source-run and checkpoint selection used when loading a prior model.",
    )

    @model_validator(mode="after")
    def _validate_cross_section_rules(self):
        return _validate_cross_section_consistency(self, emit_warnings=True)
