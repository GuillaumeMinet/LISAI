from __future__ import annotations

from typing import Literal, TypeAlias

Activation2D: TypeAlias = Literal["ReLU", "swish"]
ActivationLVAE: TypeAlias = Literal["relu", "leakyrelu", "elu", "selu"]
NormType: TypeAlias = Literal["group", "batch"] | None
UpsamplingMethod2D: TypeAlias = Literal["conv", "nearest", "bilinear", "bicubic"]
UpsamplingMethod3D: TypeAlias = Literal["conv", "nearest", "trilinear"]
UpsamplingOrder: TypeAlias = Literal["before", "after"]
UpsamplingNet: TypeAlias = Literal["unet", "rcan"]
RCANUpsamplingMethod: TypeAlias = Literal["conv", "pixelshuffle"]
