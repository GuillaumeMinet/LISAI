import torch
import torch.nn as nn
import math
import warnings
import os, sys

sys.path.append(os.getcwd() + './')
from lisai.models.common import swish,upsamp_block_3d,downsamp_block_3d,conv_block_3d,get_timestep_embedding


class UNet_PosEncod(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.norm = kwargs.get("norm")
        self.gr_norm = kwargs.get("gr_norm")
        self.dropout = kwargs.get("dropout")
        self.depth = kwargs.get("depth")
        self.feat = kwargs.get("feat")
        self.upsampling_factor = kwargs.get("upsampling_factor",1)
        self.upsampling_order = kwargs.get("upsampling_order")
        self.upsampling_method = kwargs.get("upsampling_method")
        self.remove_skip_con = kwargs.get("remove_skip_con",False)
        self.ch = kwargs.get("ch", 128)
        self.temb_ch = self.ch * 4
        self.in_channels = kwargs.get("in_channels")
        self.out_channels = kwargs.get("out_channels")

        if kwargs.get("activation") == "ReLU":
            self.activation = torch.nn.ReLU()
        elif kwargs.get("activation") == "swish":
            self.activation = swish
        else:
            raise Exception("Activation should be 'ReLU' or 'swish'.")

        # Encoder
        self.encoder_conv = nn.ModuleList()
        self.encoder_downsamp = nn.ModuleList()
        for level in range(self.depth):
            if level == 0:
                in_ch = self.in_channels
                out_ch = self.feat
            else:
                in_ch = self.feat * 2 ** (level - 1)
                out_ch = self.feat * 2 ** level
            conv_b = conv_block_3d(in_ch, out_ch, activation=self.activation,
                                temb_channels=self.temb_ch, norm=self.norm,
                                gr_norm=self.gr_norm, dropout=self.dropout)
            self.encoder_conv.append(conv_b)
            self.encoder_downsamp.append(downsamp_block_3d(out_ch, out_ch, activation=self.activation,
                                                        stride=(1, 2, 2), norm=self.norm,kernel_size=(1,2,2),
                                                        gr_norm=self.gr_norm,
                                                        dropout=self.dropout))

        # Base conv block
        self.base = conv_block_3d(self.feat * (2 ** (self.depth - 1)), self.feat * 2 ** self.depth,
                               activation=self.activation, temb_channels=self.temb_ch,
                               norm=self.norm, gr_norm=self.gr_norm, dropout=self.dropout)

        # Decoder
        self.decoder_upsamp = nn.ModuleList()
        self.decoder_conv = nn.ModuleList()
        for level in range(0,self.depth):
            in_ch = self.feat * 2**(level + 1)
            out_ch = self.feat * 2**(level)
            upsamp = upsamp_block_3d(in_ch, out_ch, activation=self.activation, norm=self.norm,stride=(1,2,2),kernel_size=(1,2,2))
            self.decoder_upsamp.append(upsamp)
            if self.remove_skip_con >= level:
                in_ch = out_ch
            conv_b = conv_block_3d(in_ch, out_ch, activation=self.activation, temb_channels=self.temb_ch,
                                norm=self.norm, gr_norm=self.gr_norm, dropout=self.dropout)
            self.decoder_conv.append(conv_b)

        # Output conv
        self.out_conv = nn.Conv3d(self.feat, self.out_channels, 1)

        # Upsampling layers
        if self.upsampling_method == 'conv':
            self.upsamp_before = upsamp_block_3d(self.in_channels, self.in_channels, activation=self.activation,
                                              kernel_size=(1,2,2), stride=(1,2,2),
                                              norm=self.norm, gr_norm=1, dropout=self.dropout)
        else:
            self.upsamp_before = nn.Upsample(scale_factor=self.upsampling_factor, mode=self.upsampling_method)

        self.upsamp_after = upsamp_block_3d(self.feat, self.feat, activation=self.activation,
                                         kernel_size=(1,2,2), stride=(1,2,2),
                                         norm=self.norm, gr_norm=self.gr_norm, dropout=self.dropout)
        self.upsamp_after_conv = conv_block_3d(self.feat, self.feat, activation=self.activation,
                                            norm=self.norm, gr_norm=self.gr_norm, dropout=self.dropout)

        # Timestep embedding
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(self.ch, self.temb_ch),
            torch.nn.Linear(self.temb_ch, self.temb_ch),
        ])

    def forward(self, x, t=None):
        if t is not None:
            temb = get_timestep_embedding(t, self.ch)
            temb = self.temb.dense[0](temb)
            temb = self.activation(temb)
            temb = self.temb.dense[1](temb)
        else:
            temb = None

        # print(x.shape)
        if self.upsampling_factor > 1 and self.upsampling_order == 'before':
            x = self.upsamp_before(x)

        encoder_out = []
        for level in range(self.depth):
            # print(level)
            x = self.encoder_conv[level](x, temb)
            encoder_out.append(x)
            x = self.encoder_downsamp[level](x)
            # print(x.shape)

        x = self.base(x, temb)
        # print(x.shape)
        for level in range(self.depth - 1, -1, -1):
            # print(level,x.shape)
            # print(self.decoder_upsamp[level])
            x = self.decoder_upsamp[level](x)
            if self.remove_skip_con >= level:
                x = self.decoder_conv[level](x, temb)
            else:
                x = self.decoder_conv[level](torch.cat((x, encoder_out[level]), dim=1), temb)

        if self.upsampling_factor > 1 and self.upsampling_order == 'after':
            x = self.upsamp_after(x)
            x = self.upsamp_after_conv(x)
        
        x = self.out_conv(x)
        return x
