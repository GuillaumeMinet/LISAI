from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from lisai.models.params import LVAEParams, RCANParams, UNet3DParams, UNetParams, UNetRCANParams


class UNetModelSection(BaseModel):
    """Typed model section for the standalone 2D U-Net architecture."""

    model_config = ConfigDict(extra="forbid")

    architecture: Literal["unet"] = Field(
        default="unet",
        description="Select the standalone 2D U-Net architecture.",
    )
    parameters: UNetParams = Field(
        default_factory=UNetParams,
        description="Typed constructor parameters for the standalone 2D U-Net model.",
    )


class UNet3DModelSection(BaseModel):
    """Typed model section for the standalone 3D U-Net architecture."""

    model_config = ConfigDict(extra="forbid")

    architecture: Literal["unet3d"] = Field(
        default="unet3d",
        description="Select the standalone 3D U-Net architecture.",
    )
    parameters: UNet3DParams = Field(
        default_factory=UNet3DParams,
        description="Typed constructor parameters for the standalone 3D U-Net model.",
    )


class RCANModelSection(BaseModel):
    """Typed model section for the standalone RCAN architecture."""

    model_config = ConfigDict(extra="forbid")

    architecture: Literal["rcan"] = Field(
        default="rcan",
        description="Select the standalone RCAN architecture.",
    )
    parameters: RCANParams = Field(
        default_factory=RCANParams,
        description="Typed constructor parameters for the standalone RCAN model.",
    )


class UNetRCANModelSection(BaseModel):
    """Typed model section for the hybrid UNet-RCAN architecture."""

    model_config = ConfigDict(extra="forbid")

    architecture: Literal["unet_rcan"] = Field(
        default="unet_rcan",
        description="Select the hybrid UNet-RCAN architecture.",
    )
    parameters: UNetRCANParams = Field(
        default_factory=UNetRCANParams,
        description="Typed constructor parameters for the hybrid UNet-RCAN model.",
    )


class LVAEModelSection(BaseModel):
    """Typed model section for LadderVAE."""

    model_config = ConfigDict(extra="forbid")

    architecture: Literal["lvae"] = Field(
        default="lvae",
        description="Select the LadderVAE architecture.",
    )
    parameters: LVAEParams = Field(
        default_factory=LVAEParams,
        description="Typed constructor parameters for LadderVAE.",
    )


ModelSection = Annotated[
    UNetModelSection | UNet3DModelSection | RCANModelSection | UNetRCANModelSection | LVAEModelSection,
    Field(discriminator="architecture"),
]

__all__ = [
    "LVAEModelSection",
    "ModelSection",
    "RCANModelSection",
    "UNet3DModelSection",
    "UNetModelSection",
    "UNetRCANModelSection",
]
