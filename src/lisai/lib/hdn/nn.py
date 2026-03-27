import torch
from torch import nn


def _closest_divisor(value: int, target: int) -> int:
    divisors = [d for d in range(1, value + 1) if value % d == 0]
    return min(divisors, key=lambda d: abs(d - target))


def _make_norm_layer(norm: str | None, *, channels: int, gr_norm: int):
    if norm is None:
        return None
    if norm == "batch":
        return nn.BatchNorm2d(channels)
    if norm == "group":
        if gr_norm <= 0:
            raise ValueError("`gr_norm` must be > 0 when `norm='group'`.")
        groups = gr_norm if channels % gr_norm == 0 else _closest_divisor(channels, gr_norm)
        return nn.GroupNorm(num_groups=groups, num_channels=channels)
    raise ValueError(f"unrecognized norm type '{norm}'")


class ResidualBlock(nn.Module):
    """
    Residual block with 2 convolutional layers.
    Input, intermediate, and output channels are the same. Padding is always
    'same'. The 2 convolutional layers have the same groups. No stride allowed,
    and kernel sizes have to be odd.

    The result is:
        out = gate(f(x)) + x
    where an argument controls the presence of the gating mechanism, and f(x)
    has different structures depending on the argument block_type.
    block_type is a string specifying the structure of the block, where:
        a = activation
        b = normalization layer
        c = conv layer
        d = dropout.
    For example, bacdbacd has 2x (norm, activation, conv, dropout).
    """

    default_kernel_size = (3, 3)

    def __init__(self,
                 channels,
                 nonlin,
                 kernel=None,
                 groups=1,
                 norm="group",
                 gr_norm=8,
                 block_type=None,
                 dropout=None,
                 gated=None):
        super().__init__()
        if kernel is None:
            kernel = self.default_kernel_size
        elif isinstance(kernel, int):
            kernel = (kernel, kernel)
        elif len(kernel) != 2:
            raise ValueError(
                "kernel has to be None, int, or an iterable of length 2")
        assert all([k % 2 == 1 for k in kernel]), "kernel sizes have to be odd"
        kernel = list(kernel)
        pad = [k // 2 for k in kernel]
        self.gated = gated

        modules = []

        if block_type == 'cabdcabd':
            for i in range(2):
                conv = nn.Conv2d(channels,
                                 channels,
                                 kernel[i],
                                 padding=pad[i],
                                 groups=groups)
                modules.append(conv)
                modules.append(nonlin())
                norm_layer = _make_norm_layer(norm, channels=channels, gr_norm=gr_norm)
                if norm_layer is not None:
                    modules.append(norm_layer)
                if dropout is not None:
                    modules.append(nn.Dropout2d(dropout))

        elif block_type == 'bacdbac':
            for i in range(2):
                norm_layer = _make_norm_layer(norm, channels=channels, gr_norm=gr_norm)
                if norm_layer is not None:
                    modules.append(norm_layer)
                modules.append(nonlin())
                conv = nn.Conv2d(channels,
                                 channels,
                                 kernel[i],
                                 padding=pad[i],
                                 groups=groups)
                modules.append(conv)
                if dropout is not None and i == 0:
                    modules.append(nn.Dropout2d(dropout))

        elif block_type == 'bacdbacd':
            for i in range(2):
                norm_layer = _make_norm_layer(norm, channels=channels, gr_norm=gr_norm)
                if norm_layer is not None:
                    modules.append(norm_layer)
                modules.append(nonlin())
                conv = nn.Conv2d(channels,
                                 channels,
                                 kernel[i],
                                 padding=pad[i],
                                 groups=groups)
                modules.append(conv)
                modules.append(nn.Dropout2d(dropout))

        else:
            raise ValueError("unrecognized block type '{}'".format(block_type))

        if gated:
            modules.append(GateLayer2d(channels, 1, nonlin))
        self.block = nn.Sequential(*modules)

    def forward(self, x):
        return self.block(x) + x


class ResidualGatedBlock(ResidualBlock):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, gated=True)


class GateLayer2d(nn.Module):
    """
    Double the number of channels through a convolutional layer, then use
    half the channels as gate for the other half.
    """

    def __init__(self, channels, kernel_size, nonlin=nn.LeakyReLU):
        super().__init__()
        assert kernel_size % 2 == 1
        pad = kernel_size // 2
        self.conv = nn.Conv2d(channels, 2 * channels, kernel_size, padding=pad)
        self.nonlin = nonlin()

    def forward(self, x):
        x = self.conv(x)
        x, gate = torch.chunk(x, 2, dim=1)
        x = self.nonlin(x)  # TODO remove this?
        gate = torch.sigmoid(gate)
        return x * gate
