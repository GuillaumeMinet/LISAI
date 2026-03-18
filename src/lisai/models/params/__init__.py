from __future__ import annotations

from .lvae import LVAEParams
from .rcan import RCANBackboneParams, RCANParams
from .unet import UNetBackboneParams, UNetParams
from .unet3d import UNet3DParams
from .unet_rcan import UNetRCANParams

AnyModelParams = UNetParams | UNet3DParams | RCANParams | UNetRCANParams | LVAEParams

__all__ = [
    "AnyModelParams",
    "LVAEParams",
    "RCANBackboneParams",
    "RCANParams",
    "UNetBackboneParams",
    "UNetParams",
    "UNet3DParams",
    "UNetRCANParams",
]
