from .unet import UNet_PosEncod as UNet2D
from .unet3d import UNet_PosEncod as UNet3D  # We can alias it here!
from .rcan import RCAN
from .unet_rcan import UNetRCAN
from .lvae import LadderVAE

# Optional: Define the registry right here
MODEL_REGISTRY = {
    "unet": UNet2D,
    "unet3d": UNet3D,
    "rcan": RCAN,
    "unet_rcan": UNetRCAN,
    "lvae": LadderVAE,
}

__all__ = ["MODEL_REGISTRY", "UNet2D", "UNet3D", "RCAN", "UNetRCAN", "LadderVAE"]