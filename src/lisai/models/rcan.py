from __future__ import annotations

import logging

import torch.nn as nn

from lisai.models.common import RG, conv_block, upsamp_block
from lisai.models.params import RCANParams

logger = logging.getLogger("RCAN")


class RCAN(nn.Module):
    def __init__(self, params: RCANParams):
        super().__init__()
        self.params = params
        self.upsamp = params.upsamp
        self.upsamp_stride = params.upsamp_stride if self.upsamp is not None else None
        self.upsamp_pad = self.upsamp_stride - 1 if self.upsamp_stride is not None else None

        self.sf = nn.Conv2d(params.in_channels, params.num_features, kernel_size=3, padding=1)
        self.rgs = nn.Sequential(
            *[RG(params.num_features, params.num_rcab, params.reduction, params.dropout) for _ in range(params.num_rg)]
        )
        self.conv1 = nn.Conv2d(params.num_features, params.num_features, kernel_size=3, padding=1)

        if params.collapse_ch_before_upsamp:
            self.conv2 = nn.Conv2d(params.num_features, params.out_channels, kernel_size=3, padding=1)
            upsamp_ch = params.out_channels
        else:
            self.conv2 = nn.Conv2d(params.num_features, params.num_features, kernel_size=3, padding=1)
            upsamp_ch = params.num_features

        if self.upsamp is not None:
            if params.upsampling_method == "pixelshuffle":
                self.upsamp_block = nn.Sequential(
                    nn.Conv2d(params.num_features, 4 * params.num_features, kernel_size=3, padding=1),
                    nn.PixelShuffle(2),
                )
            elif params.upsampling_method == "conv":
                kernel_size = self.upsamp * params.upsamp_kernel_factor
                upsamp_pad = params.upsamp_kernel_factor - 1
                self.upsamp_block = nn.Sequential(
                    upsamp_block(
                        upsamp_ch,
                        upsamp_ch,
                        kernel_size=kernel_size,
                        stride=self.upsamp,
                        padding=upsamp_pad,
                        norm=None,
                        dropout=0,
                    ),
                    conv_block(upsamp_ch, params.out_channels, norm=None, dropout=0),
                )
            else:
                raise ValueError(
                    f"Upsampling method should be 'conv' or 'pixelshuffle' but got {params.upsampling_method} instead."
                )
        else:
            self.upsamp_block = None

    def forward(self, x, *args):
        x = self.sf(x)
        residual = x
        x = self.rgs(x)
        x = self.conv1(x)
        x += residual
        x = self.conv2(x)
        if self.upsamp_block is not None:
            x = self.upsamp_block(x)
        return x
