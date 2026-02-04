import torch 
import torch.nn as nn
import logging
import os,sys

logger = logging.getLogger("Model UNetRCAN")

sys.path.append(os.getcwd() + './')
from lisai.models.rcan import RCAN
from lisai.models.unet import UNet_PosEncod as UNet
from lisai.models.common import upsamp_block,conv_block

class UNetRCAN(nn.Module):
    """
    UNet + RCAN implementation.
    """
    def __init__(self, UNet_prm:dict,RCAN_prm:dict,upsampling_factor:int=1,upsampling_net="unet",**kwargs):
        
        super().__init__()
        
        self.upsampling_factor = upsampling_factor
        self.upsampling_net = upsampling_net 
        
        if self.upsampling_factor > 1:
            assert upsampling_net in ["unet","rcan"], "upsampling net should be unet or rcan"

            if self.upsampling_net == "unet":
                RCAN_prm["upsamp"] = None
                upsamp_method = UNet_prm.get("upsampling_method","conv")
                if upsamp_method == "conv":
                    ch=UNet_prm.get("in_channels")
                    self.upsampblock = nn.Sequential(
                        upsamp_block(ch,ch,kernel_size=self.upsampling_factor, stride=self.upsampling_factor,norm = UNet_prm.get("norm"),gr_norm = 1,dropout = 0),
                        conv_block(ch,ch,norm = UNet_prm.get("norm"),gr_norm = 1,dropout = 0))
                else:
                    self.upsampblock = nn.Upsample(scale_factor = self.upsampling_factor,mode=self.upsampling_method)

            elif self.upsampling_net == "rcan":
                self.upsampblock = None
                RCAN_prm["upsamp"] = self.upsampling_factor
        else:
            RCAN_prm["upsamp"] = None

        if RCAN_prm.get("in_channels") is None:
            RCAN_prm["in_channels"] = UNet_prm.get("in_channels") + UNet_prm.get("out_channels")

        self.unet = UNet(**UNet_prm)
        self.rcan = RCAN(**RCAN_prm)

    def forward(self, x,*args):
        if self.upsampling_factor > 1 and self.upsampling_net == "unet":
            x = self.upsampblock(x)   
        out_unet = self.unet(x)
        input_rcan = torch.cat([out_unet, x],dim=1)
        out_rcan = self.rcan(input_rcan) 
        return out_rcan
