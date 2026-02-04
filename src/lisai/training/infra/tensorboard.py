from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from lisai.lib.utils import get_paths
import torch

def handle_tensorboard(CFG):
    if not CFG.get("tensorboard", {}).get("saving", False):
        return None

    dt = datetime.now().strftime("%d-%m-%Y_%H-%M-%S_")
    path = get_paths.get_tensorboard_path(
        dataset_name=CFG["dataset_name"],
        subfolder=CFG["tensorboard_prm"].get("subfolder", "")
    )
    return SummaryWriter(log_dir=str(path / (dt + CFG["exp_name"])))


def log_images_to_tensorboard(writer, x, y, prediction, epoch, volumetric=False):
    if writer is None:
        return

    shape = "CHW" if volumetric else "HW"

    def normalize(img):
        img = img.cpu().detach()
        img = img - torch.min(img)
        img = 255 * img / torch.max(img)
        return img.type(torch.uint8)

    writer.add_image("input", normalize(x[0, 0]), epoch, dataformats=shape)
    writer.add_image("prediction", normalize(prediction[0, 0]), epoch, dataformats=shape)
    if y is not None:
        writer.add_image("ground truth", normalize(y[0, 0]), epoch, dataformats=shape)

