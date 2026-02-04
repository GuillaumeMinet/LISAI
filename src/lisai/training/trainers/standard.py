import torch
import numpy as np
from tqdm import tqdm
from lisai.training.losses import get_loss_function
from lisai.training.infra import tensorboard, validation_images
from .base import BaseTrainer

class StandardTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Standard Loss setup
        self.loss_function = get_loss_function(**self.training_prm.get("loss_function", {}))

    def _forward_pass(self, x, y, samp_pos):
        prediction = self.model(x, samp_pos)
        if self.loss_function is None:
            raise ValueError("Loss function undefined")
        
        loss_val = self.loss_function(prediction, y)
        return loss_val, prediction

    def train_epoch(self, epoch):
        self.model.train()
        train_loss = []
        
        iter_loader = self.train_loader
        if self.pbar:
            iter_loader = tqdm(iter_loader, leave=False, position=0, desc=f'Train Ep {epoch}')

        for batch_id, batch in enumerate(iter_loader):
            virtual_batches = self._split_batch(batch)
            
            for vb in zip(*virtual_batches):
                x, y, samp_pos = self._prepare_batch(vb)
                loss_val, _ = self._forward_pass(x, y, samp_pos)

                train_loss.append(loss_val.item())
                
                loss_val.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                if self.update_console:
                    self.logger.info(f"Epoch {epoch} | Batch {batch_id}/{len(self.train_loader)}")

        return {'loss': np.mean(train_loss)}

    def validate(self, epoch, save_imgs=False):
        self.model.eval()
        val_loss = []
        val_imgs = []

        iter_loader = self.val_loader
        if self.pbar:
            iter_loader = tqdm(iter_loader, leave=False, position=0, desc=f'Val Ep {epoch}')

        with torch.no_grad():
            for batch in iter_loader:
                virtual_batches = self._split_batch(batch)
                
                for vb in zip(*virtual_batches):
                    x, y, samp_pos = self._prepare_batch(vb)
                    loss_val, pred = self._forward_pass(x, y, samp_pos)
                    
                    val_loss.append(loss_val.item())

                    if self.writer:
                         tensorboard.log_images_to_tensorboard(self.writer, vb[0], vb[1], pred, epoch, self.volumetric)

                    if save_imgs:
                        val_imgs.append((vb[0], vb[1], pred))

        if save_imgs:
             validation_images.save_validation_images(val_imgs, self.validation_images_folder, self.volumetric)

        return {'loss': np.mean(val_loss)}

    def log_headers(self):
        return ["Train_loss", "Val_loss"]

    def log_values(self, train_metrics, val_metrics):
        return [f"{train_metrics['loss']:.5f}", f"{val_metrics['loss']:.5f}"]