from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ColorCodeDefaults(BaseModel):
    model_config = ConfigDict(extra="forbid")

    colormap: str = "turbo"
    saturation: float = 0.35
    add_colorbar: bool = True
    zstep: float = 0.4


class ColorCodeOverrides(BaseModel):
    model_config = ConfigDict(extra="forbid")

    colormap: str | None = None
    saturation: float | None = None
    add_colorbar: bool | None = None
    zstep: float | None = None


class ApplyDefaults(BaseModel):
    model_config = ConfigDict(extra="forbid")

    save_folder: str | None = "default"
    in_place: bool = False
    epoch_number: int | None = None
    best_or_last: str = "best"
    filters: list[str] = Field(default_factory=lambda: ["tiff", "tif"])
    skip_if_contain: list[str] | None = None
    crop_size: int | tuple[int, int] | None = None
    keep_original_shape: bool = True
    tiling_size: int | None = None
    stack_selection_idx: int | None = None
    timelapse_max: int | None = None
    lvae_num_samples: int | None = 20
    lvae_save_samples: bool = True
    denormalize_output: bool = True
    save_inp: bool = False
    downsamp: int | None = None
    apply_color_code: bool = False
    color_code_prm: ColorCodeDefaults = Field(default_factory=ColorCodeDefaults)
    dark_frame_context_length: bool = False


class ApplyOverrides(BaseModel):
    model_config = ConfigDict(extra="forbid")

    save_folder: str | None = None
    in_place: bool | None = None
    epoch_number: int | None = None
    best_or_last: str | None = None
    filters: list[str] | None = None
    skip_if_contain: list[str] | None = None
    crop_size: int | tuple[int, int] | None = None
    keep_original_shape: bool | None = None
    tiling_size: int | None = None
    stack_selection_idx: int | None = None
    timelapse_max: int | None = None
    lvae_num_samples: int | None = None
    lvae_save_samples: bool | None = None
    denormalize_output: bool | None = None
    save_inp: bool | None = None
    downsamp: int | None = None
    apply_color_code: bool | None = None
    color_code_prm: ColorCodeOverrides | None = None
    dark_frame_context_length: bool | None = None


class EvaluateDefaults(BaseModel):
    model_config = ConfigDict(extra="forbid")

    best_or_last: str = "best"
    epoch_number: int | None = None
    test_loader: Any | None = None
    tiling_size: int | None = None
    crop_size: int | tuple[int, int] | None = None
    metrics_list: list[str] | None = None
    lvae_num_samples: int | None = 20
    results: dict[str, Any] | None = None
    save_folder: str | None = None
    overwrite: bool = False
    eval_gt: str | None = None
    data_prm_update: dict[str, Any] | None = None
    ch_out: int | None = None
    split: str = "test"
    limit_n_imgs: int | None = None


class EvaluateOverrides(BaseModel):
    model_config = ConfigDict(extra="forbid")

    best_or_last: str | None = None
    epoch_number: int | None = None
    tiling_size: int | None = None
    crop_size: int | tuple[int, int] | None = None
    metrics_list: list[str] | None = None
    lvae_num_samples: int | None = None
    save_folder: str | None = None
    overwrite: bool | None = None
    eval_gt: str | None = None
    data_prm_update: dict[str, Any] | None = None
    ch_out: int | None = None
    split: str | None = None
    limit_n_imgs: int | None = None


class InferenceDefaults(BaseModel):
    model_config = ConfigDict(extra="forbid")

    apply: ApplyDefaults = Field(default_factory=ApplyDefaults)
    evaluate: EvaluateDefaults = Field(default_factory=EvaluateDefaults)


class InferenceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    apply: ApplyOverrides | None = None
    evaluate: EvaluateOverrides | None = None
