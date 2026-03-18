from __future__ import annotations

import logging
import warnings

import torch
import torch.nn as nn

from lisai.models.common import (
    conv_block,
    downsamp_block,
    get_timestep_embedding,
    swish,
    upsamp_block,
)
from lisai.models.params import UNetParams

logger = logging.getLogger("UNet")


def _activation_from_name(name: str):
    if name == "ReLU":
        return torch.nn.ReLU()
    if name == "swish":
        return swish
    raise ValueError("Activation should be 'ReLU' or 'swish'.")


class UNet_PosEncod(nn.Module):
    """U-Net implementation with optional learned upsampling."""

    def __init__(self, params: UNetParams):
        super().__init__()
        self.params = params
        self.norm = params.norm
        self.gr_norm = params.gr_norm
        self.dropout = params.dropout
        self.depth = params.depth
        self.feat = params.feat
        self.upsampling_factor = params.upsampling_factor
        self.upsampling_order = params.upsampling_order
        self.upsampling_method = params.upsampling_method
        self.remove_skip_con = params.remove_skip_con
        self.cab_skip_con = params.cab_skip_con
        self.ch = params.ch
        self.temb_ch = self.ch * 4
        self.in_channels = params.in_channels
        self.out_channels = params.out_channels
        self.activation = _activation_from_name(params.activation)

        if self.cab_skip_con:
            warnings.warn(
                "`cab_skip_con` is accepted but not implemented in UNet_PosEncod yet; using plain skip connections instead.",
                stacklevel=2,
            )

        self.encoder_conv = nn.ModuleList()
        self.encoder_downsamp = nn.ModuleList()
        for level in range(self.depth):
            if level == 0:
                in_ch = self.in_channels
                out_ch = self.feat
            else:
                in_ch = self.feat * 2 ** (level - 1)
                out_ch = self.feat * 2 ** level
            conv_b = conv_block(
                in_ch,
                out_ch,
                activation=self.activation,
                temb_channels=self.temb_ch,
                norm=self.norm,
                gr_norm=self.gr_norm,
                dropout=self.dropout,
            )
            self.encoder_conv.append(conv_b)
            self.encoder_downsamp.append(
                downsamp_block(
                    out_ch,
                    out_ch,
                    activation=self.activation,
                    stride=2,
                    norm=self.norm,
                    gr_norm=self.gr_norm,
                    dropout=self.dropout,
                )
            )

        self.base = conv_block(
            self.feat * (2 ** (self.depth - 1)),
            self.feat * 2 ** self.depth,
            activation=self.activation,
            temb_channels=self.temb_ch,
            norm=self.norm,
            gr_norm=self.gr_norm,
            dropout=self.dropout,
        )

        self.decoder_upsamp = nn.ModuleList()
        self.decoder_conv = nn.ModuleList()
        for level in range(self.depth):
            in_ch = self.feat * 2 ** (level + 1)
            out_ch = self.feat * 2 ** level
            self.decoder_upsamp.append(
                upsamp_block(in_ch, out_ch, activation=self.activation, norm=self.norm)
            )

            decoder_in_ch = out_ch if self._skip_connection_removed(level) else in_ch
            self.decoder_conv.append(
                conv_block(
                    decoder_in_ch,
                    out_ch,
                    activation=self.activation,
                    temb_channels=self.temb_ch,
                    norm=self.norm,
                    gr_norm=self.gr_norm,
                    dropout=self.dropout,
                )
            )

        self.out_conv = nn.Conv2d(self.feat, self.out_channels, 1)

        if self.upsampling_method == "conv":
            self.upsamp_before = upsamp_block(
                self.in_channels,
                self.in_channels,
                activation=self.activation,
                kernel_size=self.upsampling_factor,
                stride=self.upsampling_factor,
                norm=self.norm,
                gr_norm=1,
                dropout=self.dropout,
            )
        else:
            self.upsamp_before = nn.Upsample(
                scale_factor=self.upsampling_factor,
                mode=self.upsampling_method,
            )

        self.upsamp_after = upsamp_block(
            self.feat,
            self.feat,
            activation=self.activation,
            kernel_size=self.upsampling_factor,
            stride=self.upsampling_factor,
            norm=self.norm,
            gr_norm=self.gr_norm,
            dropout=self.dropout,
        )
        self.upsamp_after_conv = conv_block(
            self.feat,
            self.feat,
            activation=self.activation,
            norm=self.norm,
            gr_norm=self.gr_norm,
            dropout=self.dropout,
        )

        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList(
            [
                torch.nn.Linear(self.ch, self.temb_ch),
                torch.nn.Linear(self.temb_ch, self.temb_ch),
            ]
        )

    def _skip_connection_removed(self, level: int) -> bool:
        return self.remove_skip_con > 0 and level < self.remove_skip_con

    def forward(self, x, t=None):
        if t is not None:
            temb = get_timestep_embedding(t, self.ch)
            temb = self.temb.dense[0](temb)
            temb = self.activation(temb)
            temb = self.temb.dense[1](temb)
        else:
            temb = None

        if self.upsampling_factor > 1 and self.upsampling_order == "before":
            x = self.upsamp_before(x)

        encoder_out = []
        for level in range(self.depth):
            x = self.encoder_conv[level](x)
            encoder_out.append(x)
            x = self.encoder_downsamp[level](x)

        x = self.base(x)

        for level in range(self.depth - 1, -1, -1):
            x = self.decoder_upsamp[level](x)
            if self._skip_connection_removed(level):
                x = self.decoder_conv[level](x)
            else:
                x = self.decoder_conv[level](torch.cat((x, encoder_out[level]), dim=1))

        if self.upsampling_factor > 1 and self.upsampling_order == "after":
            x = self.upsamp_after(x)
            x = self.upsamp_after_conv(x)

        x = self.out_conv(x)
        return x
