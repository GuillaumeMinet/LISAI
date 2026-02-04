import torch
import numpy as np
from tqdm import tqdm
from lisai.training.infra import tensorboard, validation_images
from lisai.lib.hdn.forwardpass import forward_pass as lvae_forward_pass
from lisai.lib.hdn.forwardpass import forward_pass_tiling as lvae_forward_pass_tiling
from .base import BaseTrainer

class LVAETrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.betaKL = self.training_prm.get("betaKL", 1.0)
        self._init_tiling()

    def _init_tiling(self):
        self.tiling_validation = False
        if self.data_prm.get("val_patch_size") is not None:
            if self.data_prm.get("val_patch_size") != self.data_prm.get("patch_size"):
                self.tiling_validation = True
                self.tiling_patch = self.data_prm.get("patch_size")
                if self.data_prm.get("downsampling", {}).get("downsamp_factor"):
                    self.tiling_patch //= self.data_prm.get("downsampling").get("downsamp_factor")

    def _run_lvae_pass(self, x, y, validation=False):
        """Helper to handle tiling vs standard LVAE pass"""
        if validation and self.tiling_validation:
            outputs = lvae_forward_pass_tiling(
                x, None, self.device, self.model,
                patch_size=getattr(self, "tiling_patch", None)
            )
        else:
            outputs = lvae_forward_pass(
                x, y if not validation else None,
                self.device, self.model
            )
        
        recons = outputs["recons_loss"]
        kl = outputs["kl_loss"]
        total_loss = recons + self.betaKL * kl
        return total_loss, kl, recons, outputs["out_mean"]

    def train_epoch(self, epoch):
        self.model.train()
        losses = {'loss': [], 'kl': [], 'recons': []}
        
        iter_loader = self.train_loader
        if self.pbar:
            iter_loader = tqdm(iter_loader, leave=False, position=0, desc=f'Train Ep {epoch}')

        for batch_id, batch in enumerate(iter_loader):
            virtual_batches = batching.create_virtual_batches(batch, self.batch_size)
            
            for vb in zip(*virtual_batches):
                x, y, samp_pos = self._prepare_batch(vb) # inherited from Base
                
                loss_val, kl, recons, _ = self._run_lvae_pass(x, y, validation=False)
                
                losses['loss'].append(loss_val.item())
                losses['kl'].append(kl.item())
                losses['recons'].append(recons.item())

                loss_val.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
        
        return {k: np.mean(v) for k, v in losses.items()}

    def validate(self, epoch, save_imgs=False):
        self.model.eval()
        losses = {'loss': [], 'kl': [], 'recons': []}
        val_imgs = []

        iter_loader = self.val_loader
        if self.pbar:
            iter_loader = tqdm(iter_loader, leave=False, position=0, desc=f'Val Ep {epoch}')

        with torch.no_grad():
            for batch in iter_loader:
                virtual_batches = batching.create_virtual_batches(batch, self.batch_size)
                
                for vb in zip(*virtual_batches):
                    x, y, samp_pos = self._prepare_batch(vb)
                    loss_val, kl, recons, pred = self._run_lvae_pass(x, y, validation=True)

                    losses['loss'].append(loss_val.item())
                    losses['kl'].append(kl.item())
                    losses['recons'].append(recons.item())

                    if self.writer:
                         tensorboard.log_images_to_tensorboard(self.writer, vb[0], vb[1], pred, epoch, self.volumetric)

                    if save_imgs:
                        val_imgs.append((vb[0], vb[1], pred))

        if save_imgs:
             validation_images.save_validation_images(val_imgs, self.validation_images_folder, self.volumetric)

        return {k: np.mean(v) for k, v in losses.items()}

    def log_headers(self):
        return ["Train_loss", "Val_loss", "Recons", "KL"]

    def log_values(self, train_metrics, val_metrics):
        # Specific formatting for LVAE logs
        return [
            f"{train_metrics['loss']:.5f}", 
            f"{val_metrics['loss']:.5f}",
            f"{train_metrics['recons']:.5f}", 
            f"{train_metrics['kl']:.5f}"
        ]