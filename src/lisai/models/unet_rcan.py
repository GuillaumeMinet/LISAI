from __future__ import annotations

import logging

import torch
import torch.nn as nn

from lisai.models.common import conv_block, upsamp_block
from lisai.models.params import UNetRCANParams
from lisai.models.rcan import RCAN
from lisai.models.unet import UNet_PosEncod as UNet

logger = logging.getLogger("Model UNetRCAN")


class UNetRCAN(nn.Module):
    """UNet + RCAN implementation."""

    def __init__(self, params: UNetRCANParams):
        super().__init__()
        self.params = params
        self.upsampling_factor = params.upsampling_factor
        self.upsampling_net = params.upsampling_net

        unet_params = params.resolved_unet_params()
        rcan_params = params.resolved_rcan_params()

        if self.upsampling_factor > 1 and self.upsampling_net == "unet":
            if unet_params.upsampling_method == "conv":
                ch = unet_params.in_channels
                self.upsampblock = nn.Sequential(
                    upsamp_block(
                        ch,
                        ch,
                        kernel_size=self.upsampling_factor,
                        stride=self.upsampling_factor,
                        norm=unet_params.norm,
                        gr_norm=1,
                        dropout=0,
                    ),
                    conv_block(ch, ch, norm=unet_params.norm, gr_norm=1, dropout=0),
                )
            else:
                self.upsampblock = nn.Upsample(
                    scale_factor=self.upsampling_factor,
                    mode=unet_params.upsampling_method,
                )
        else:
            self.upsampblock = None

        self.unet = UNet(unet_params)
        self.rcan = RCAN(rcan_params)

    def forward(self, x, *args):
        if self.upsampling_factor > 1 and self.upsampling_net == "unet":
            x = self.upsampblock(x)
        out_unet = self.unet(x)
        input_rcan = torch.cat([out_unet, x], dim=1)
        out_rcan = self.rcan(input_rcan)
        return out_rcan
