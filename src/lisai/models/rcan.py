import torch 
import torch.nn as nn
import logging
import os,sys


logger = logging.getLogger("RCAN")

sys.path.append(os.getcwd() + './')
from lisai.models.common import conv_block,upsamp_block,RG


class RCAN(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        in_channels = kwargs.get("in_channels")
        out_channels = kwargs.get("out_channels",1)
        num_features = kwargs.get("num_features",1)
        num_rg = kwargs.get("num_rg",1)
        num_rcab = kwargs.get("num_rcab",1)
        reduction = kwargs.get("reduction",1)
        dropout = kwargs.get("dropout",0)
        self.upsamp = kwargs.get("upsamp",None)
        if self.upsamp is not None:
            self.upsamp_stride = kwargs.get("upsamp_stride",1)
            self.upsamp_pad = self.upsamp_stride-1
        self.sf = nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1)
        self.rgs = nn.Sequential(*[RG(num_features, num_rcab, reduction,dropout) for _ in range(num_rg)])
        self.conv1 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(num_features, out_channels, kernel_size=3, padding=1)

        if self.upsamp is not None:
            upsamp_method = kwargs.get("upsampling_method", "conv")
            if upsamp_method == "pixelshuffle":
                assert self.upsamp == 2, "upsamp > 2 with pixelshuffle not implemented"
                self.upsamp_block = nn.Sequential(
                    nn.Conv2d(num_features,4*num_features,kernel_size=3,padding=1),
                    nn.PixelShuffle(2)
                )
            elif upsamp_method == "conv":
                upsamp_kernel_factor = kwargs.get("upsamp_kernel_factor",1)
                kernel_size = self.upsamp*upsamp_kernel_factor
                upsamp_pad = upsamp_kernel_factor-1
                self.upsamp_block = nn.Sequential(
                        upsamp_block(1,1,kernel_size=kernel_size, stride=self.upsamp,padding=upsamp_pad,norm = None,dropout = 0),
                        conv_block(1,1,norm = None,dropout = 0))
            else:
                raise ValueError (f"Upsampling method should be 'conv' or 'pixelshuffle' but got {upsamp_method} instead.")
        else:
            self.upsamp_block = None



    def forward(self, x,*args):
        x = self.sf(x)
        residual = x
        x = self.rgs(x)
        x = self.conv1(x)
        x += residual
        x = self.conv2(x)
        if self.upsamp_block is not None:
            x = self.upsamp_block(x)
        return x