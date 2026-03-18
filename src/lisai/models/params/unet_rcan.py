from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .common import UpsamplingNet
from .rcan import RCANBackboneParams, RCANParams
from .unet import UNetBackboneParams, UNetParams


class UNetRCANParams(BaseModel):
    """Typed constructor parameters for the hybrid UNet-RCAN architecture."""

    model_config = ConfigDict(extra="forbid")

    upsampling_net: UpsamplingNet = Field(
        default="unet",
        description="Which sub-network performs learned upsampling when upsampling_factor > 1.",
    )
    upsampling_factor: int = Field(
        default=1,
        description="Global upsampling factor performed by the hybrid model.",
    )
    UNet_prm: UNetBackboneParams = Field(
        default_factory=UNetBackboneParams,
        description="Typed U-Net backbone parameters used inside the hybrid model.",
    )
    RCAN_prm: RCANBackboneParams = Field(
        default_factory=RCANBackboneParams,
        description="Typed RCAN backbone parameters used inside the hybrid model.",
    )

    @field_validator("upsampling_factor")
    @classmethod
    def _validate_upsampling_factor(cls, value: int):
        if value <= 0:
            raise ValueError("`upsampling_factor` must be > 0.")
        return value

    @model_validator(mode="after")
    def _validate_nested_params(self):
        if self.UNet_prm.cab_skip_con:
            # Accepted but not implemented in the U-Net body. The actual runtime warning
            # is emitted from the model constructor so the user sees it during execution too.
            pass
        return self

    def resolved_unet_params(self) -> UNetParams:
        return UNetParams.model_validate(self.UNet_prm.model_dump())

    def resolved_rcan_params(self) -> RCANParams:
        derived_in_channels = self.RCAN_prm.in_channels
        if derived_in_channels is None:
            derived_in_channels = self.UNet_prm.in_channels + self.UNet_prm.out_channels

        derived_upsamp = None
        if self.upsampling_factor > 1 and self.upsampling_net == "rcan":
            derived_upsamp = self.upsampling_factor

        return RCANParams.model_validate(
            self.RCAN_prm.model_dump()
            | {
                "in_channels": derived_in_channels,
                "upsamp": derived_upsamp,
            }
        )

    def effective_upsampling_factor(self) -> int:
        return int(self.upsampling_factor)
