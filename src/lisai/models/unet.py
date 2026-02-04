import torch 
import torch.nn as nn
import math
import warnings
import os,sys
import logging

logger = logging.getLogger("UNet")

sys.path.append(os.getcwd() + './')
from lisai.models.common import swish,upsamp_block,downsamp_block,conv_block,get_timestep_embedding


class UNet_PosEncod(nn.Module):
    """
    U-Net implementation with optional upsampling and optional positional encoding.
    Arguments:
      self.in_channels: int, number of input channels
      self.out_channels: int, number of output channels
      self.depth: int, number of downsampling levels.
      !to be completed!
    """

    def __init__(self,**kwargs):
        
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
        self.ch = kwargs.get("ch",128)
        self.temb_ch = self.ch*4
        self.in_channels = kwargs.get("in_channels")
        self.out_channels = kwargs.get("out_channels")

        # activation definition
        if kwargs.get("activation") == "ReLU":
            self.activation = torch.nn.ReLU()
        elif kwargs.get("activation") == "swish": 
            self.activation = swish
        else:
            raise Exception("Activation should be 'ReLU' or 'swish'.")

        # encoder
        self.encoder_conv = nn.ModuleList()
        self.encoder_downsamp = nn.ModuleList()
        for level in range(self.depth): #e.g. self.depth = 3, creating levels: 0,1,2, and level 3 will be the "base" one below.
            if level == 0: 
                in_ch = self.in_channels
                out_ch = self.feat
            else:
                in_ch = self.feat * 2**(level-1)
                out_ch = self.feat * 2**(level) 
            conv_b = conv_block(in_ch,out_ch,activation = self.activation,
                                temb_channels=self.temb_ch,norm = self.norm,
                                gr_norm = self.gr_norm, dropout = self.dropout)
            self.encoder_conv.append(conv_b)
            self.encoder_downsamp.append(downsamp_block(out_ch,out_ch,activation=self.activation,
                                                        stride = 2,norm = self.norm,
                                                        gr_norm = self.gr_norm,
                                                        dropout = self.dropout))


        #base conv block
        self.base = conv_block(self.feat * (2 ** (self.depth-1)), self.feat* 2 ** (self.depth),
                               activation = self.activation,temb_channels=self.temb_ch,
                               norm = self.norm,gr_norm = self.gr_norm,dropout = self.dropout)
            
        #decoder
        self.decoder_upsamp = nn.ModuleList() 
        self.decoder_conv = nn.ModuleList()
        for level in range(0,self.depth): #e.g. self.depth = 3, level 3 is the base and decoder takes care of level: 2,1,0.
            in_ch = self.feat * 2**(level + 1)
            out_ch = self.feat * 2**(level)
            upsamp = upsamp_block (in_ch,out_ch,activation=self.activation,norm=self.norm)
            self.decoder_upsamp.append(upsamp)

            if self.remove_skip_con >= level:
                # If we don't have the skip connections, the convolutional block following
                # the upsampling will have the same number of features in and out
                in_ch = out_ch 
            conv_b = conv_block(in_ch,out_ch,activation = self.activation,temb_channels=self.temb_ch,
                                norm = self.norm,gr_norm = self.gr_norm,dropout = self.dropout)
            self.decoder_conv.append(conv_b)

        # output conv (used only if upsampling before)
        self.out_conv = nn.Conv2d(self.feat, self.out_channels, 1)
        

        #upsampling before
        if self.upsampling_method == 'conv':
            self.upsamp_before = upsamp_block(self.in_channels, self.in_channels, activation=self.activation,
                                               kernel_size=self.upsampling_factor, stride=self.upsampling_factor,
                                               norm = self.norm,gr_norm = 1,dropout = self.dropout)
        else:
            self.upsamp_before = nn.Upsample(scale_factor = self.upsampling_factor,mode=self.upsampling_method)
            
        #upsampling after (+ convolutional block)
        self.upsamp_after = upsamp_block(self.feat,self.feat,activation=self.activation, 
                                         kernel_size=self.upsampling_factor,stride=self.upsampling_factor,
                                         norm = self.norm,gr_norm = self.gr_norm,dropout = self.dropout)
        
        self.upsamp_after_conv = conv_block(self.feat,self.feat,activation = self.activation,
                                            norm = self.norm,gr_norm = self.gr_norm,dropout = self.dropout)

        # timestep embedding
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(self.ch,
                            self.temb_ch),
            torch.nn.Linear(self.temb_ch,
                            self.temb_ch),
        ])


    def forward(self, x, t=None):

        if t is not None:
            temb = get_timestep_embedding(t, self.ch)
            temb = self.temb.dense[0](temb)
            temb = self.activation(temb)
            temb = self.temb.dense[1](temb)
        else:
            temb = None

        if self.upsampling_factor > 1 and self.upsampling_order == 'before':
            x = self.upsamp_before(x)

        encoder_out = [] # for skip connections
        for level in range(self.depth): #e.g. self.depth = 3, going through levels 0,1,2
            x = self.encoder_conv[level](x)
            encoder_out.append(x)
            x = self.encoder_downsamp[level](x)

        x = self.base(x) #e.g. self.depth = 3, base is level 3.

        for level in range(self.depth-1,-1,-1): #e.g. self.depth = 3, going through level 2->0
            x = self.decoder_upsamp[level](x)
            if self.remove_skip_con >= level:
                x = self.decoder_conv[level](x)
            else:
                x = self.decoder_conv[level](torch.cat((x, encoder_out[level]), dim=1))

        if self.upsampling_factor > 1 and self.upsampling_order == 'after':
            x = self.upsamp_after(x)
            x = self.upsamp_after_conv(x)
       
        x = self.out_conv(x)
            
        return x
    

