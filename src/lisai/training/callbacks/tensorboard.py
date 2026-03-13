# src/lisai/training/callbacks/tensorboard.py

import torch

from .base import Callback


class TensorBoardCallback(Callback):
    def __init__(self, writer, *, volumetric: bool = False):
        self.writer = writer
        self.volumetric = volumetric

    def on_epoch_end(self, trainer, epoch: int, logs: dict):
        if self.writer is None:
            return
        self.writer.add_scalar("Train_loss", logs["train_loss"], epoch)
        self.writer.add_scalar("Val_loss", logs["val_loss"], epoch)

    def on_validation_batch_end(self, trainer, epoch: int, x, y, prediction):
        if self.writer is None:
            return
        if y is None:
            return

        # legacy: assumes x,y,pred are [B,1,...] after volumetric/channel handling
        input_img = x[0, 0, ...].detach().cpu()
        gt_img = y[0, 0, ...].detach().cpu()
        pred_img = prediction[0, 0, ...].detach().cpu()

        input_img = _to_uint8(input_img)
        gt_img = _to_uint8(gt_img)
        pred_img = _to_uint8(pred_img)

        shape = "CHW" if self.volumetric else "HW"
        self.writer.add_image("input", input_img, epoch, dataformats=shape)
        self.writer.add_image("prediction", pred_img, epoch, dataformats=shape)
        self.writer.add_image("ground truth", gt_img, epoch, dataformats=shape)


def _to_uint8(img: torch.Tensor) -> torch.Tensor:
    img = img - torch.min(img)
    mx = torch.max(img)
    if mx > 0:
        img = 255 * img / mx
    return img.to(torch.uint8)
