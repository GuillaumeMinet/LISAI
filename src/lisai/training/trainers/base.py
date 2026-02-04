import os
import time
import logging
import torch
import numpy as np
from abc import ABC, abstractmethod
from tqdm import tqdm

from lisai.training.core import optim
from lisai.training.infra import saving, tensorboard, validation_images
from lisai.lib.utils.logger_utils import CustomStreamHandler, EnableFilter

_loss_file_name = "loss.txt"
_log_file_name = "train.log"

class BaseTrainer(ABC):
    def __init__(self, model, train_loader, val_loader, device,
                 training_prm=None, data_prm=None, saving_prm=None,
                 exp_name=None, mode="train", writer=None, state_dict=None):

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.writer = writer
        self.state_dict = state_dict or {"epoch": 0}

        # Configs
        self.exp_name = exp_name
        self.mode = mode
        self.training_prm = training_prm or {}
        self.data_prm = data_prm or {}
        self.saving_prm = saving_prm or {}

        # Common Init
        self._init_training_prm()
        self._init_data_prm()
        self._init_saving_prm()
        self._init_logger()

        # Optimizer
        self.optimizer, self.scheduler = self.configure_optimizers()

    def _init_training_prm(self):
        self.batch_size = self.training_prm.get("batch_size", 8)
        self.n_epochs = self.training_prm.get("n_epochs", 100)
        self.early_stop = self.training_prm.get("early_stop", False)
        self.pos_encod = self.training_prm.get("pos_encod", False)
        
        # TQDM handling
        try:
            from tqdm import tqdm
            self.tqdm_available = True
        except ImportError:
            self.tqdm_available = False
        
        self.pbar = self.training_prm.get("pbar", False) and self.tqdm_available
        self.update_console = not self.pbar

    def _init_data_prm(self):
        self.volumetric = self.data_prm.get("volumetric", False)

    def _init_saving_prm(self):
        self.saving = self.saving_prm.get("saving", False)
        if self.saving:
            self.model_save_folder = self.saving_prm.get("model_save_folder")
            self.loss_path = self.model_save_folder / _loss_file_name
            self.log_path = self.model_save_folder / _log_file_name
            self.save_validation_images = self.saving_prm.get("save_validation_images", False)
            if self.save_validation_images:
                self.save_validation_freq = self.saving_prm.get("save_validation_freq", 10)
                self.validation_images_folder = self.model_save_folder / "validation_images"
                os.makedirs(self.validation_images_folder, exist_ok=True)
        else:
            self.loss_path = None
            self.log_path = None
            self.save_validation_images = False

    def _init_logger(self):
        self.logger = logging.getLogger(f"lisai.trainer.{self.exp_name}")

    # --- Abstract Methods (To be implemented by subclasses) ---
    @abstractmethod
    def train_epoch(self, epoch):
        """Returns dict of metrics, e.g. {'loss': 0.5, 'kl': 0.1}"""
        pass

    @abstractmethod
    def validate(self, epoch, save_imgs=False):
        """Returns dict of metrics"""
        pass
    
    @abstractmethod
    def log_headers(self):
        """Returns list of column names for loss.txt"""
        pass

    @abstractmethod
    def log_values(self, train_metrics, val_metrics):
        """Returns list of values corresponding to headers"""
        pass

    # --- Main Loop ---
    def train(self):
        start_epoch = self.state_dict.get("epoch", 0)
        best_loss = self.state_dict.get("best_loss", float('inf'))

        iter_epoch = range(start_epoch, self.n_epochs)
        if self.pbar:
            iter_epoch = tqdm(iter_epoch, position=1, total=self.n_epochs, initial=start_epoch)
            iter_epoch.set_description('Epochs')

        self.logger.info(f"Experiment: {self.exp_name}, Mode: {self.mode}, Device: {self.device}")

        for epoch in iter_epoch:
            try:
                save_val_imgs = self.save_validation_images and (epoch % self.save_validation_freq == 0)
                
                # Run Epochs
                train_metrics = self.train_epoch(epoch)
                val_metrics = self.validate(epoch, save_imgs=save_val_imgs)
                
                train_loss = train_metrics['loss']
                val_loss = val_metrics['loss']

                # Update State Dict
                self.state_dict.update({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict()
                })
                # Scheduler
                if self.scheduler:
                    self.state_dict["scheduler"] = self.scheduler.state_dict()
                    if self.training_prm.get("scheduler") == "ReduceLROnPlateau":
                        self.scheduler.step(val_loss)
                    else:
                        self.scheduler.step()

                # Best Model Check
                is_best = val_loss < best_loss
                if is_best:
                    best_loss = val_loss
                    self.logger.info(f"epoch {epoch}: Best model saved with best_val = {best_loss:.5f}.")

                # Saving
                if self.saving:
                    self._handle_saving(is_best)
                    self._update_loss_file(epoch, train_metrics, val_metrics)

                # Tensorboard
                if self.writer:
                    self.writer.add_scalar('Train_loss', train_loss, epoch)
                    self.writer.add_scalar('Val_loss', val_loss, epoch)

                # Early Stop
                if self.early_stop and epoch > 2:
                    # Note: Real early stop logic usually requires patience, 
                    # strictly implementing what was in the original code for now.
                    self.logger.info("Early stopping triggered.")
                    break

            except KeyboardInterrupt:
                self.logger.info(f"Training manually stopped at epoch {epoch}.")
                return
            except Exception as e:
                self.logger.error(f"Training stopped due to error: {e}")
                raise

    def _handle_saving(self, is_best):
        epoch = self.state_dict["epoch"]
        folder = self.model_save_folder
        
        # Helper to reduce boilerplate
        def save_ckpt(name, obj):
            torch.save(obj, folder / name)

        if self.saving_prm.get("state_dict", True):
            save_ckpt("model_last_state_dict.pt", self.state_dict)
            if is_best:
                name = "model_best_state_dict.pt" if self.saving_prm.get("overwrite_best", True) else f"model_epoch_{epoch}_state_dict.pt"
                save_ckpt(name, self.state_dict)

        if self.saving_prm.get("entire_model", True):
            save_ckpt("model_last.pt", self.model)
            if is_best:
                name = "model_best.pt" if self.saving_prm.get("overwrite_best", True) else f"model_epoch_{epoch}.pt"
                save_ckpt(name, self.model)

    def _update_loss_file(self, epoch, train_metrics, val_metrics):
        if not os.path.exists(self.loss_path):
            with open(self.loss_path, 'w') as f:
                headers = ["Epoch"] + self.log_headers()
                f.write(f"{' '.join([f'{h:<15}' for h in headers])}\n")

        with open(self.loss_path, 'a') as f:
            values = [epoch] + self.log_values(train_metrics, val_metrics)
            # Simple formatter
            f.write(f"{' '.join([f'{v:<15}' for v in values])}\n")

    def _prepare_batch(self, batch):
        x, y, *samp_pos = batch
        
        # Handle NaN y
        if torch.isnan(y).all().item():
            y = None
        else:
            y = y.to(self.device)

        if self.pos_encod and samp_pos:
            samp_pos = samp_pos[0].to(self.device)
        else:
            samp_pos = None

        if len(x.shape) == 4 and self.volumetric:
            x = x.unsqueeze(1)
            if y is not None:
                y = y.unsqueeze(1)

        x = x.to(self.device, dtype=torch.float)
        return x, y, samp_pos
    
    def _split_batch(self, batch):
        """
        Splits a batch into virtual batches for gradient accumulation.
        Handles the specific tuple structure (x, y, *pos).
        """
        # If batch is smaller than virtual batch size, return it as-is
        if int(batch[0].shape[0] // self.batch_size) == 0:
            return [(tensor,) for tensor in batch]
            
        return [torch.split(tensor, self.batch_size, dim=0) for tensor in batch]
    
    def configure_optimizers(self):
        """
        Sets up optimizer and scheduler based on training_prm.
        Can be overridden by subclasses (e.g. if multiple optimizers).
        """
        lr = self.training_prm.get("lr", 1e-3) # Good to have a default
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        scheduler_str = self.training_prm.get("scheduler")
        scheduler = None
        
        if scheduler_str == "linear":
            scheduler = torch.optim.lr_scheduler.LinearLR(optimizer)
        elif scheduler_str == "ReduceLROnPlateau":
             scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 'min', patience=10, factor=0.5, min_lr=1e-12
            )
        elif scheduler_str:
            raise ValueError(f"Scheduler {scheduler_str} unknown.")

        return optimizer, scheduler