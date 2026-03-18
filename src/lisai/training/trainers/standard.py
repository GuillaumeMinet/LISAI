# src/lisai/training/trainers/standard.py

import numpy as np
import torch

from .base import BaseTrainer

try:
    from tqdm import tqdm
except Exception:
    pass


class StandardTrainer(BaseTrainer):
    @property
    def is_lvae(self) -> bool:
        return False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # config-driven custom loss function
        loss_cfg_model = self.cfg.loss_function

        if loss_cfg_model is not None:
            try:
                from lisai.training.losses import get_loss_function

                self.loss_function = get_loss_function(**loss_cfg_model.as_kwargs())
            except Exception:
                self.logger.error(f"Invalid loss_function config: {loss_cfg_model}", exc_info=True)
                raise
        else:
            self.loss_function = torch.nn.MSELoss()
            self.logger.warning("Loss function config not found, falling back to MSE loss")

    def train_epoch(self, epoch: int) -> dict:
        self.model.train()
        self.model.to(self.device)
        losses = []

        iter_loader = self.train_loader
        if self.pbar:
            iter_loader = tqdm(iter_loader, leave=False, position=0)
            iter_loader.set_description(f"Training - Epoch {epoch}")

        for batch_id, batch in enumerate(iter_loader):
            if self.update_console:
                self._update_console_new_batch(epoch, batch_id, len(iter_loader))

            virtual_batches = self._split_batch(batch, warn_once=(batch_id == 0))

            for (x, y, *samp_pos) in zip(*virtual_batches):
                samp_pos = samp_pos[0] if (self.pos_encod and len(samp_pos) == 1) else None
                x, y, samp_pos = self._prepare_batch(x, y, samp_pos)

                pred = self.model(x, samp_pos)
                loss = self.loss_function(pred, y)

                loss.backward()
                losses.append(loss.item())

            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.early_stop is not False and batch_id > 0:
                break

        return {"loss": float(np.mean(losses))}

    def validate(self, epoch: int, save_imgs: bool = False) -> dict:
        self.model.eval()
        self.model.to(self.device)
        losses = []

        val_imgs = [] if save_imgs else None

        with torch.no_grad():
            iter_val = self.val_loader
            if self.pbar:
                from tqdm import tqdm

                iter_val = tqdm(iter_val, position=0, leave=False)
                iter_val.set_description(f"Validation - Epoch {epoch}")

            for batch in iter_val:
                virtual_batches = self._split_batch(batch)

                for (x, y, *samp_pos) in zip(*virtual_batches):
                    samp_pos = samp_pos[0] if (self.pos_encod and len(samp_pos) == 1) else None
                    x, y, samp_pos = self._prepare_batch(x, y, samp_pos)

                    pred = self.model(x, samp_pos)
                    losses.append(self.loss_function(pred, y).item())

                    # tensorboard images hook (legacy behavior)
                    for cb in self.callbacks:
                        cb.on_validation_batch_end(self, epoch, x, y, pred)

                    if save_imgs:
                        for i in range(x.shape[0]):
                            if y is None:
                                val_imgs.append((x[i], None, pred[i]))
                            else:
                                val_imgs.append((x[i], y[i], pred[i]))

        if save_imgs:
            for cb in self.callbacks:
                cb.on_validation_images_end(self, epoch, val_imgs)

        return {"loss": float(np.mean(losses))}
