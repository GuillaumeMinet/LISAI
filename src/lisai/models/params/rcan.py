from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .common import RCANUpsamplingMethod


class RCANBackboneParams(BaseModel):
    """Typed backbone parameters for RCAN blocks embedded in larger architectures."""

    model_config = ConfigDict(extra="forbid")

    in_channels: int | None = Field(
        default=None,
        description="Number of input channels. When omitted inside UNet-RCAN, it is derived automatically.",
    )
    out_channels: int = Field(default=1, description="Number of output channels produced by RCAN.")
    num_features: int = Field(default=64, description="Feature-channel width used inside the RCAN body.")
    num_rg: int = Field(default=8, description="Number of residual groups.")
    num_rcab: int = Field(default=12, description="Number of RCAB blocks per residual group.")
    reduction: int = Field(default=16, description="Channel-attention reduction factor.")
    dropout: float = Field(default=0.0, description="Dropout probability used inside RCAN residual blocks.")
    upsamp_kernel_factor: int = Field(
        default=1,
        description="Extra multiplier applied to the transposed-convolution kernel when RCAN performs upsampling.",
    )
    upsampling_method: RCANUpsamplingMethod = Field(
        default="conv",
        description="RCAN upsampling method when standalone or when UNet-RCAN delegates upsampling to RCAN.",
    )
    collapse_ch_before_upsamp: bool = Field(
        default=True,
        strict=True,
        description="Whether RCAN should collapse features to out_channels before running the upsampling head.",
    )

    @field_validator("out_channels", "num_features", "num_rg", "num_rcab", "reduction", "upsamp_kernel_factor")
    @classmethod
    def _validate_positive_ints(cls, value: int):
        if value <= 0:
            raise ValueError("RCAN channel and depth parameters must be > 0.")
        return value

    @field_validator("in_channels")
    @classmethod
    def _validate_optional_in_channels(cls, value: int | None):
        if value is not None and value <= 0:
            raise ValueError("`in_channels` must be > 0 when provided.")
        return value

    @field_validator("dropout")
    @classmethod
    def _validate_dropout(cls, value: float):
        if value < 0 or value >= 1:
            raise ValueError("`dropout` must be in the interval [0, 1).")
        return value


class RCANParams(RCANBackboneParams):
    """Typed constructor parameters for the standalone RCAN model."""

    in_channels: int = Field(default=1, description="Number of input channels expected by the standalone RCAN model.")
    upsamp: int | None = Field(default=None, description="Optional standalone upsampling factor performed by RCAN.")
    upsamp_stride: int = Field(default=1, description="Stride used by the RCAN upsampling block when upsamp is enabled.")

    @field_validator("upsamp", "upsamp_stride")
    @classmethod
    def _validate_optional_positive_ints(cls, value: int | None):
        if value is not None and value <= 0:
            raise ValueError("RCAN upsampling parameters must be > 0 when provided.")
        return value

    @model_validator(mode="after")
    def _validate_upsampling_settings(self):
        if self.upsampling_method == "pixelshuffle":
            if self.upsamp is None:
                raise ValueError("`upsamp` is required when `upsampling_method='pixelshuffle'`.")
            if self.upsamp != 2:
                raise ValueError("`pixelshuffle` upsampling is only implemented for `upsamp=2`.")
        return self

    def effective_upsampling_factor(self) -> int:
        return int(self.upsamp) if self.upsamp is not None else 1
