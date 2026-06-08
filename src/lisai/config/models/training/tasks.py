from __future__ import annotations

from copy import deepcopy
from typing import Annotated, Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator, model_validator

from .loss import BasicLossName, CharEdgeLossName


TaskName = Literal[
    "custom",
    "upsamp_single_frame",
    "upsamp_multiframes",
    "denoising_hdn",
    "denoising_care",
    "denoising_unetrcan",
]
DenoisingLossName = BasicLossName | CharEdgeLossName


class CustomTaskSection(BaseModel):
    """No-op task preset. Low-level config values are used as authored."""

    model_config = ConfigDict(extra="allow")

    name: Literal["custom"] = Field(
        default="custom",
        description="Do not apply task-specific config overrides.",
    )


class UpsampSingleFrameTaskSection(BaseModel):
    """Single-frame upsampling task using multiple sparse samples as input channels."""

    model_config = ConfigDict(extra="forbid")

    name: Literal["upsamp_single_frame"] = Field(
        default="upsamp_single_frame",
        description="Single-frame upsampling with multiple sparse input samples.",
    )
    upsampling_factor: int = Field(
        description="Spatial upsampling factor. Also used as the data downsampling factor.",
    )
    sampling_ratio: Literal[0.25, 0.5, 0.75] = Field(
        description="Fraction of sparse samples kept for multiple-sampling input generation.",
    )

    @field_validator("upsampling_factor")
    @classmethod
    def _validate_upsampling_factor(cls, value: int) -> int:
        if value < 2:
            raise ValueError("`upsampling_factor` must be >= 2.")
        return value

    @model_validator(mode="after")
    def _validate_multiple_channel_count(self):
        _multiple_sampling_channels(
            upsampling_factor=self.upsampling_factor,
            sampling_ratio=float(self.sampling_ratio),
        )
        return self


class UpsampMultipleSamplingTaskSection(BaseModel):
    """Temporal-window upsampling task using frame context as input channels."""

    model_config = ConfigDict(extra="forbid")

    name: Literal["upsamp_multiframes"] = Field(
        default="upsamp_multiframes",
        description="Upsampling with a temporal context window packed into input channels.",
    )
    upsampling_factor: int = Field(
        description="Spatial upsampling factor. Also used as the data downsampling factor.",
    )
    temporal_window: int = Field(
        description="Odd temporal context window size. This overrides data.timelapse_prm.context_length.",
    )

    @field_validator("upsampling_factor")
    @classmethod
    def _validate_upsampling_factor(cls, value: int) -> int:
        if value < 2:
            raise ValueError("`upsampling_factor` must be >= 2.")
        return value

    @field_validator("temporal_window")
    @classmethod
    def _validate_temporal_window(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("`temporal_window` must be > 0.")
        if value % 2 == 0:
            raise ValueError("`temporal_window` must be odd.")
        return value


class DenoisingHDNTaskSection(BaseModel):
    """HDN/LVAE denoising task."""

    model_config = ConfigDict(extra="forbid")

    name: Literal["denoising_hdn"] = Field(
        default="denoising_hdn",
        description="HDN/LVAE denoising. Requires an LVAE model and a task-level betaKL value.",
    )
    betaKL: float = Field(
        description="Weight applied to the LVAE KL loss term.",
    )
    supervised: bool = Field(
        description="Whether the HDN task expects an explicitly paired target dataset.",
    )

    @field_validator("betaKL")
    @classmethod
    def _validate_beta_kl(cls, value: float) -> float:
        if value < 0:
            raise ValueError("`betaKL` must be >= 0.")
        return value


class DenoisingCARETaskSection(BaseModel):
    """Supervised U-Net denoising task."""

    model_config = ConfigDict(extra="forbid")

    name: Literal["denoising_care"] = Field(
        default="denoising_care",
        description="Supervised CARE-style U-Net denoising.",
    )
    loss: DenoisingLossName = Field(
        description="Loss used for supervised CARE-style denoising.",
    )


class DenoisingUNetRCANTaskSection(BaseModel):
    """Supervised UNet-RCAN denoising task."""

    model_config = ConfigDict(extra="forbid")

    name: Literal["denoising_unetrcan"] = Field(
        default="denoising_unetrcan",
        description="Supervised UNet-RCAN denoising.",
    )
    loss: DenoisingLossName = Field(
        description="Loss used for supervised UNet-RCAN denoising.",
    )


ExperimentTaskSection = Annotated[
    CustomTaskSection
    | UpsampSingleFrameTaskSection
    | UpsampMultipleSamplingTaskSection
    | DenoisingHDNTaskSection
    | DenoisingCARETaskSection
    | DenoisingUNetRCANTaskSection,
    Field(discriminator="name"),
]

_TASK_ADAPTER = TypeAdapter(ExperimentTaskSection)


def normalize_task_value(value: Any) -> Any:
    """Normalize shorthand task YAML before Pydantic discriminates by name."""
    if value is None:
        return {"name": "custom"}
    if isinstance(value, str):
        return {"name": value}
    return value


def parse_task_value(value: Any) -> ExperimentTaskSection:
    return _TASK_ADAPTER.validate_python(normalize_task_value(value))


def apply_experiment_task_overrides(value: Any) -> Any:
    """Expand high-level task presets into low-level data/model config fields."""
    if not isinstance(value, dict):
        return value

    experiment = value.get("experiment")
    if not isinstance(experiment, Mapping):
        return value

    task = parse_task_value(experiment.get("task"))
    if task.name == "custom":
        return value

    cfg = deepcopy(value)
    if isinstance(task, UpsampSingleFrameTaskSection):
        _apply_single_frame_task(cfg, task)
    elif isinstance(task, UpsampMultipleSamplingTaskSection):
        _apply_temporal_window_task(cfg, task)
    elif isinstance(task, DenoisingHDNTaskSection):
        _apply_hdn_denoising_task(cfg, task)
    elif isinstance(task, (DenoisingCARETaskSection, DenoisingUNetRCANTaskSection)):
        _apply_supervised_denoising_task(cfg, task)
    return cfg


def _multiple_sampling_channels(*, upsampling_factor: int, sampling_ratio: float) -> int:
    raw_channels = upsampling_factor**2 * sampling_ratio
    n_channels = round(raw_channels)
    if n_channels < 1 or abs(raw_channels - n_channels) > 1e-9:
        raise ValueError(
            "`upsampling_factor**2 * sampling_ratio` must yield a positive integer input-channel count."
        )
    return int(n_channels)


def _section(config: dict[str, Any], key: str) -> dict[str, Any]:
    section = config.get(key)
    if not isinstance(section, dict):
        section = {}
        config[key] = section
    return section


def _apply_single_frame_task(config: dict[str, Any], task: UpsampSingleFrameTaskSection) -> None:
    data = _section(config, "data")
    downsampling = _section(data, "downsampling")
    n_channels = _multiple_sampling_channels(
        upsampling_factor=task.upsampling_factor,
        sampling_ratio=float(task.sampling_ratio),
    )

    if "timelapse_prm" in data or data.get("data_format") == "timelapse":
        timelapse_prm = data.get("timelapse_prm")
        if not isinstance(timelapse_prm, dict):
            timelapse_prm = {}
            data["timelapse_prm"] = timelapse_prm
        timelapse_prm["context_length"] = None

    downsampling["downsamp_factor"] = int(task.upsampling_factor)
    downsampling["downsamp_method"] = "multiple"
    downsampling["multiple_prm"] = {
        "fill_factor": float(task.sampling_ratio),
        "random": False,
    }

    _apply_model_geometry(
        config,
        input_channels=n_channels,
        upsampling_factor=int(task.upsampling_factor),
    )


def _apply_temporal_window_task(config: dict[str, Any], task: UpsampMultipleSamplingTaskSection) -> None:
    data = _section(config, "data")
    timelapse_prm = data.get("timelapse_prm")
    if not isinstance(timelapse_prm, dict):
        timelapse_prm = {}
        data["timelapse_prm"] = timelapse_prm
    timelapse_prm["context_length"] = int(task.temporal_window)

    downsampling = _section(data, "downsampling")
    downsampling["downsamp_factor"] = int(task.upsampling_factor)
    if downsampling.get("downsamp_method") == "multiple":
        downsampling["downsamp_method"] = "random"
    downsampling.setdefault("downsamp_method", "random")
    downsampling.pop("multiple_prm", None)

    _apply_model_geometry(
        config,
        input_channels=int(task.temporal_window),
        upsampling_factor=int(task.upsampling_factor),
    )


def _apply_model_geometry(
    config: dict[str, Any],
    *,
    input_channels: int,
    upsampling_factor: int,
) -> None:
    model = config.get("model")
    if not isinstance(model, dict):
        return

    architecture = model.get("architecture")
    parameters = _section(model, "parameters")

    if architecture == "unet_rcan":
        parameters["upsampling_factor"] = int(upsampling_factor)
        unet_prm = _section(parameters, "UNet_prm")
        unet_prm["in_channels"] = int(input_channels)
        unet_prm["out_channels"] = int(input_channels)

        rcan_prm = _section(parameters, "RCAN_prm")
        rcan_prm["in_channels"] = int(input_channels) * 2
        rcan_prm["out_channels"] = 1
        return

    if architecture in {"unet", "unet3d"}:
        parameters["in_channels"] = int(input_channels)
        parameters["out_channels"] = 1
        parameters["upsampling_factor"] = int(upsampling_factor)
        parameters.setdefault("upsampling_order", "after")
        return

    if architecture == "rcan":
        parameters["in_channels"] = int(input_channels)
        parameters["out_channels"] = 1
        parameters["upsamp"] = int(upsampling_factor)


def _apply_hdn_denoising_task(config: dict[str, Any], task: DenoisingHDNTaskSection) -> None:
    training = _section(config, "training")
    training["betaKL"] = float(task.betaKL)


def _apply_supervised_denoising_task(
    config: dict[str, Any],
    task: DenoisingCARETaskSection | DenoisingUNetRCANTaskSection,
) -> None:
    raw_loss_function = config.get("loss_function")
    config["loss_function"] = _loss_config_from_task_loss(task.loss, raw_loss_function)


def _loss_config_from_task_loss(loss_name: DenoisingLossName, raw_loss_function: Any) -> dict[str, Any]:
    raw_params = raw_loss_function if isinstance(raw_loss_function, Mapping) else {}
    requested_name = raw_params.get("name")
    if requested_name is not None and _loss_family(str(requested_name)) != _loss_family(str(loss_name)):
        raise ValueError(
            "`loss_function.name` cannot override `experiment.task.loss`; "
            f"got loss_function.name={requested_name!r} and experiment.task.loss={loss_name!r}."
        )

    if loss_name in {"CharEdge_loss", "CharEdge"}:
        invalid_param_blocks = {"MSE_upsampling_prm"} & set(raw_params)
        if invalid_param_blocks:
            keys = ", ".join(sorted(invalid_param_blocks))
            raise ValueError(f"`{keys}` is not valid when `experiment.task.loss` is CharEdge.")

        char_edge_params = raw_params.get("CharEdge_loss_prm")
        if char_edge_params is None:
            char_edge_params = {}
        if not isinstance(char_edge_params, Mapping):
            raise ValueError("`loss_function.CharEdge_loss_prm` must be an object.")

        return {
            "name": loss_name,
            "CharEdge_loss_prm": {
                "alpha": char_edge_params.get("alpha", 0.05),
            },
        }

    invalid_param_blocks = {"CharEdge_loss_prm", "MSE_upsampling_prm"} & set(raw_params)
    if invalid_param_blocks:
        keys = ", ".join(sorted(invalid_param_blocks))
        raise ValueError(f"`{keys}` is not valid when `experiment.task.loss` is {loss_name!r}.")
    return {"name": loss_name}


def _loss_family(loss_name: str) -> str:
    if loss_name in {"MSE", "mse", "l2", "L2"}:
        return "mse"
    if loss_name in {"MAE", "mae", "l1", "L1"}:
        return "mae"
    if loss_name in {"CharEdge_loss", "CharEdge"}:
        return "charedge"
    return loss_name


__all__ = [
    "CustomTaskSection",
    "DenoisingCARETaskSection",
    "DenoisingHDNTaskSection",
    "DenoisingUNetRCANTaskSection",
    "ExperimentTaskSection",
    "TaskName",
    "UpsampMultipleSamplingTaskSection",
    "UpsampSingleFrameTaskSection",
    "apply_experiment_task_overrides",
    "normalize_task_value",
    "parse_task_value",
]
