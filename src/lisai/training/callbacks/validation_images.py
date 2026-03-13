import os
from pathlib import Path

import numpy as np
from tifffile import imread, imwrite

from .base import Callback


class ValidationImagesCallback(Callback):
    def __init__(
        self,
        images_dir: Path | None,
        *,
        enabled: bool = False,
        freq: int = 10,
        volumetric: bool = False,
    ):
        self.enabled = enabled and images_dir is not None
        self.freq = freq
        self.volumetric = volumetric

        if self.enabled:
            self.folder = images_dir
            os.makedirs(self.folder, exist_ok=True)

    def on_validation_images_end(self, trainer, epoch: int, list_imgs):
        if not self.enabled:
            return
        if epoch % self.freq != 0:
            return
        self._save_valid_images(list_imgs, epoch)

    def _save_valid_images(self, list_imgs, epoch: int):
        for i, (x, y, pred) in enumerate(list_imgs):
            paired = (y is not None)

            inp = x.cpu().numpy()
            gt = y.cpu().numpy() if paired else None
            pred = pred.detach().cpu().numpy()

            # adjust shape (legacy behavior)
            if self.volumetric:  # [C,Z,H,W] => [Z,H,W]
                inp = inp[0, ...]
                pred = pred[0, ...]
                if paired:
                    gt = gt[0, ...]

            elif pred.shape[0] > 1:  # [C,H,W]
                mltpl_ch = True
            else:  # [1,H,W] => [H,W]
                mltpl_ch = False
                pred = pred[0, ...]

            pred_path = self.folder / f"patch{i:02d}_prediction.tiff"
            inp_path = self.folder / f"patch{i:02d}_input.tiff"
            gt_path = self.folder / f"patch{i:02d}_groundtruth.tiff"

            if os.path.exists(pred_path):
                prev = imread(pred_path)
                if len(prev.shape) == 2:
                    prev = np.expand_dims(prev, axis=0)
                elif len(prev.shape) == 3 and (self.volumetric or mltpl_ch):
                    prev = np.expand_dims(prev, axis=0)

                pred = np.expand_dims(pred, axis=0)
                tosave = np.concatenate(([prev, pred]), axis=0)

                shape = "TZYX" if len(tosave.shape) == 4 else "TYX"
                imwrite(pred_path, tosave, imagej=True, metadata={"axes": shape})

            else:
                shape = "TYX" if len(pred.shape) == 3 else "YX"
                imwrite(pred_path, pred, imagej=True, metadata={"axes": shape})

                # save inp
                if inp.shape[0] > 1:
                    _inp = inp
                    shape = "TYX"
                else:
                    _inp = inp[0]
                    shape = "YX"
                imwrite(inp_path, _inp, imagej=True, metadata={"axes": shape})

                # save gt
                if paired:
                    if inp.shape[0] > 1:
                        shape = "TYX"
                    else:
                        gt = gt[0, ...]
                        shape = "YX"
                    imwrite(gt_path, gt, imagej=True, metadata={"axes": shape})
