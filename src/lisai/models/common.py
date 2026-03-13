"""
Common building blocks for unet,unet3d, and rcan.
"""

import logging
import math

import torch
import torch.nn as nn

logger = logging.getLogger("model common blocks")

def _getDivisors(n) : 
    divisors = []
    i = 1
    while i <= n : 
        if (n % i==0) : 
            divisors.append(i), 
        i = i + 1
    return divisors

def _find_closest_divisor(a,b,smaller=False):
    """
    Rounds "b" to the closest divisor making it a divisor of "a".
    If smaller = True, it forces a value < b.
    """
    if a % b == 0:
        return b
    all  = _getDivisors(a)
    for idx,val in enumerate(all):
        if b < val:
            if idx == 0:
                return b
            if smaller or (val-b)>(b-all[idx-1]):
                return all[idx-1]
            return val

def swish(x):
    return x*torch.sigmoid(x)

def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0,1,0,0))
    return emb

def normalize(norm_type,channels,groups):
    if norm_type == 'group':
        if channels % groups != 0:
            groups = _find_closest_divisor(channels,groups)
        norm = torch.nn.GroupNorm(groups, channels)
    elif norm_type == 'batch':
        norm = torch.nn.BatchNorm2d(channels)
    else:
        logger.warning ("Normalization type unknown -> set to None.")
        norm = None
    return norm


class conv_block(nn.Module):
    """
    2 convolution layers, with optional positional encoding injected in between the 2.
    """
    def __init__(self,in_channels,out_channels,activation = swish,
                 kernel_size=3,padding=1,temb_channels = 512,
                 norm = 'group',gr_norm = 8,dropout=0.0):
        
        super().__init__()
        self.conv1=nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,padding=padding)
        self.conv2=nn.Conv2d(out_channels,out_channels,kernel_size=kernel_size,padding=padding)
        self.activation = activation
        self.temb_proj = nn.Linear(temb_channels,out_channels)
        if norm is not None:
            self.norm = normalize(norm,out_channels,gr_norm)
        else: 
            self.norm = None
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, temb=None):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.dropout(x)
        if self.norm is not None:
            x = self.norm(x)

        if temb is not None:
            x = x + self.temb_proj(self.activation(temb))[:,:,None,None]

        x = self.conv2(x)
        x = self.activation(x)
        x = self.dropout(x)
        if self.norm is not None:
            x = self.norm(x)

        return x


class conv_block_3d(nn.Module):
    """
    2 convolution layers with 3d convolutions, with optional 
    positional encoding injected in between the 2.
    NOTE: positional encoding never tested for 3d data!
    """
    def __init__(self, in_channels, out_channels, activation=swish,
                 kernel_size=3, padding=1, temb_channels=512,
                 norm='group', gr_norm=8, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.activation = activation
        self.temb_proj = nn.Linear(temb_channels, out_channels)
        self.norm = normalize(norm, out_channels, gr_norm) if norm is not None else None
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, temb=None):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.dropout(x)
        if self.norm is not None:
            x = self.norm(x)
        if temb is not None:
            x = x + self.temb_proj(self.activation(temb))[:, :, None, None, None]
        x = self.conv2(x)
        x = self.activation(x)
        x = self.dropout(x)
        if self.norm is not None:
            x = self.norm(x)
        return x




class downsamp_block(nn.Module):
    """
    Downsamples with strided convolution.
    """
    def __init__(self,in_channels,out_channels,activation = swish,kernel_size=2,
                 padding=0,stride = 2,norm = 'group',gr_norm = 8,dropout=0.0):
        super().__init__()
        self.conv=nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,padding=padding,stride = stride)
        self.activation = activation
        if norm is not None:
            self.norm = normalize(norm,out_channels,gr_norm)
        else: 
            self.norm = None
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        x = self.dropout(x)
        if self.norm is not None:
            x = self.norm(x)
        return x


class upsamp_block(nn.Module):
    """
    Upsamples with transposed convolution.
    """
    def __init__(self,in_channels,out_channels,activation = swish,kernel_size=2,
                 padding=0,stride = 2,norm = 'group',gr_norm = 8, dropout = 0.0):
        super().__init__()
        self.conv=nn.ConvTranspose2d(in_channels,out_channels,kernel_size=kernel_size,padding=padding,stride = stride)
        self.activation = activation
        if norm is not None:
            self.norm = normalize(norm,out_channels,gr_norm)
        else: 
            self.norm = None
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        x = self.dropout(x)
        if self.norm is not None:
            x = self.norm(x)
        return x



class downsamp_block_3d(nn.Module):
    """
    Downsamples with 3D strided convolution.
    """
    def __init__(self, in_channels, out_channels, activation=swish, kernel_size=2,
                 padding=0, stride=(1, 2, 2), norm='group', gr_norm=8, dropout=0.0):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)  # Changed to Conv3d
        self.activation = activation
        self.norm = normalize(norm, out_channels, gr_norm) if norm is not None else None
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        x = self.dropout(x)
        if self.norm is not None:
            x = self.norm(x)
        return x

class upsamp_block_3d(nn.Module):
    """
    Upsamples with 3D transposed convolution.
    """
    def __init__(self, in_channels, out_channels, activation=swish, kernel_size=2,
                 padding=0, stride=2, norm='group', gr_norm=8, dropout=0.0):
        super().__init__()
        self.conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)  # Changed to ConvTranspose3d
        self.activation = activation
        self.norm = normalize(norm, out_channels, gr_norm) if norm is not None else None
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        x = self.dropout(x)
        if self.norm is not None:
            x = self.norm(x)
        return x



### RCAN specific ###

class ChannelAttention(nn.Module):
    def __init__(self, num_features, reduction):
        super().__init__()
        self.module = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_features, num_features // reduction, kernel_size=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(num_features // reduction, num_features, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.module(x)


class RCAB(nn.Module):
    def __init__(self, num_features, reduction):
        super().__init__()
        self.module = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            ChannelAttention(num_features, reduction)
        )

    def forward(self, x):
        return x+self.module(x)


class RG(nn.Module):
    def __init__(self, num_features, num_rcab, reduction,dropout=0):
        super().__init__()
        self.module = [RCAB(num_features, reduction) for _ in range(num_rcab)]
        self.module.append(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1))
        self.module = nn.Sequential(*self.module)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.module(x)
        x = self.dropout(x)
        return x

