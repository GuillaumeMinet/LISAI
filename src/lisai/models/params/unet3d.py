from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .common import Activation2D, NormType, UpsamplingMethod3D, UpsamplingOrder


class UNet3DParams(BaseModel):
    """Typed constructor parameters for the 3D U-Net model."""

    model_config = ConfigDict(extra="forbid")

    feat: int = Field(default=64, description="Base number of feature channels in the first encoder level.")
    depth: int = Field(default=3, description="Number of encoder/decoder levels in the 3D U-Net.")
    in_channels: int = Field(default=1, description="Number of input channels expected by the model.")
    out_channels: int = Field(default=1, description="Number of output channels produced by the model.")
    activation: Activation2D = Field(default="swish", description="Activation function used inside the convolutional blocks.")
    norm: NormType = Field(default="group", description="Normalization type used in the convolutional blocks.")
    gr_norm: int = Field(default=8, description="Group count used when norm='group'.")
    dropout: float = Field(default=0.0, description="Dropout probability used inside convolutional blocks.")
    upsampling_factor: int = Field(default=1, description="Spatial upsampling factor applied by the standalone 3D U-Net.")
    upsampling_order: UpsamplingOrder | None = Field(
        default=None,
        description="Whether standalone upsampling happens before or after the 3D U-Net body. Required when upsampling_factor > 1.",
    )
    upsampling_method: UpsamplingMethod3D = Field(
        default="conv",
        description="Standalone upsampling method used when upsampling_factor > 1.",
    )
    remove_skip_con: int = Field(
        default=0,
        description="Number of last skip connections to remove. 0 keeps all skip connections.",
    )
    cab_skip_con: bool = Field(
        default=False,
        description="Whether CAB skip connections should be used instead of plain skip connections. Accepted for configuration but not implemented yet.",
    )
    ch: int = Field(default=128, description="Base positional-embedding channel width.")

    @field_validator("feat", "depth", "in_channels", "out_channels", "gr_norm", "ch")
    @classmethod
    def _validate_positive_ints(cls, value: int):
        if value <= 0:
            raise ValueError("Model channel and depth parameters must be > 0.")
        return value

    @field_validator("remove_skip_con")
    @classmethod
    def _validate_remove_skip_con(cls, value: int):
        if value < 0:
            raise ValueError("`remove_skip_con` must be >= 0.")
        return value

    @field_validator("dropout")
    @classmethod
    def _validate_dropout(cls, value: float):
        if value < 0 or value >= 1:
            raise ValueError("`dropout` must be in the interval [0, 1).")
        return value

    @field_validator("upsampling_factor")
    @classmethod
    def _validate_upsampling_factor(cls, value: int):
        if value <= 0:
            raise ValueError("`upsampling_factor` must be > 0.")
        return value

    @model_validator(mode="after")
    def _validate_structure(self):
        if self.remove_skip_con > self.depth:
            raise ValueError("`remove_skip_con` cannot exceed `depth`.")
        if self.upsampling_factor > 1 and self.upsampling_order is None:
            raise ValueError("`upsampling_order` is required when `upsampling_factor > 1`.")
        return self

    def effective_upsampling_factor(self) -> int:
        return int(self.upsampling_factor)
