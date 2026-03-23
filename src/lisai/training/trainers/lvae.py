# src/lisai/training/trainers/lvae.py

import numpy as np
import torch

from .base import BaseTrainer

try:
    pass
except Exception:
    pass

class LVAETrainer(BaseTrainer):
    @property
    def is_lvae(self) -> bool:
        return True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.betaKL = self.training_prm.get("betaKL", 1)

        # tiling validation logic
        self.tiling_validation = False
        self.tiling_patch = None

        val_patch_size = self.data_cfg.val_patch_size
        patch_size = self.data_cfg.patch_size

        if val_patch_size is not None and patch_size is not None and val_patch_size != patch_size:
            self.tiling_validation = True
            self.tiling_patch = patch_size
            downs = self.data_cfg.downsampling_factor
            if downs is not None:
                self.tiling_patch = self.tiling_patch // downs

        # import lvae specific forward pass
        from lisai.lib.hdn.forwardpass import forward_pass as lvae_forward_pass
        from lisai.lib.hdn.forwardpass import forward_pass_tiling as lvae_forward_pass_tiling
        self._lvae_forward_pass = lvae_forward_pass
        self._lvae_forward_pass_tiling = lvae_forward_pass_tiling

    def train_epoch(self, epoch: int) -> dict:
        self.model.train()
        self.model.to(self.device)

        losses = []
        kl_losses = []
        recons_losses = []

        iter_loader = self.train_loader
        if self.pbar:
            from tqdm import tqdm
            iter_loader = tqdm(iter_loader, leave=False, position=0)
            iter_loader.set_description(f"Training - Epoch {self._display_epoch(epoch)}")

        for batch_id, batch in enumerate(iter_loader):
            if self.update_console:
                self._update_console_new_batch(epoch,batch_id,len(iter_loader))

            virtual_batches = self._split_batch(batch, warn_once=(batch_id == 0))
            num_micro_batches = len(virtual_batches[0])
            
            for (x, y, *samp_pos) in zip(*virtual_batches):
                # LVAE ignores samp_pos in forward passes
                x, y, _ = self._prepare_batch(x, y, None)

                outputs = self._lvae_forward_pass(x, y, self.device, self.model, gaussian_noise_std=None)
                recons_loss = outputs["recons_loss"]
                kl_loss = outputs["kl_loss"]
                raw_loss = recons_loss + self.betaKL * kl_loss
                
                loss = raw_loss / num_micro_batches
                loss.backward()

                losses.append(raw_loss.item())
                kl_losses.append(kl_loss.item())
                recons_losses.append(recons_loss.item())

            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.early_stop is not False and batch_id > 0:
                break

        return {
            "loss": float(np.mean(losses)),
            "kl_loss": float(np.mean(kl_losses)),
            "recons_loss": float(np.mean(recons_losses)),
        }

    def validate(self, epoch: int, save_imgs: bool = False) -> dict:
        self.model.eval()
        self.model.to(self.device)

        losses = []
        kl_losses = []
        recons_losses = []

        val_imgs = [] if save_imgs else None

        with torch.no_grad():
            iter_val = self.val_loader
            if self.pbar:
                from tqdm import tqdm
                iter_val = tqdm(iter_val, position=0, leave=False)
                iter_val.set_description(f"Validation - Epoch {self._display_epoch(epoch)}")

            for batch in iter_val:
                virtual_batches = self._split_batch(batch)

                for (x, y, *samp_pos) in zip(*virtual_batches):
                    x, y, _ = self._prepare_batch(x, y, None)

                    if self.tiling_validation:
                        outputs = self._lvae_forward_pass_tiling(
                            x, None, self.device, self.model, gaussian_noise_std=None, patch_size=self.tiling_patch
                        )
                    else:
                        outputs = self._lvae_forward_pass(x, None, self.device, self.model, gaussian_noise_std=None)

                    recons_loss = outputs["recons_loss"]
                    kl_loss = outputs["kl_loss"]
                    loss = recons_loss + self.betaKL * kl_loss
                    prediction = outputs["out_mean"]

                    losses.append(loss.item())
                    kl_losses.append(kl_loss.item())
                    recons_losses.append(recons_loss.item())

                    for cb in self.callbacks:
                        cb.on_validation_batch_end(self, epoch, x, y, prediction)

                    if save_imgs:
                        for i in range(x.shape[0]):
                            if y is None:
                                val_imgs.append((x[i], None, prediction[i]))
                            else:
                                val_imgs.append((x[i], y[i], prediction[i]))

        if save_imgs:
            for cb in self.callbacks:
                cb.on_validation_images_end(self, epoch, val_imgs)

        return {
            "loss": float(np.mean(losses)),
            "kl_loss": float(np.mean(kl_losses)),
            "recons_loss": float(np.mean(recons_losses)),
        }
