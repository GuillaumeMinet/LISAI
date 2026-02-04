import torch
import os
import logging
import numpy as np
import time
from lisai.training.helpers import misc
from lisai.lib.utils.logger_utils import CustomStreamHandler, EnableFilter

import trainer_utils as tu

# Project-level filenames
_loss_file_name = "loss.txt"
_log_file_name = "train.log"

try:
    from tqdm import tqdm
    tqdm_available = True
except ImportError:
    tqdm_available = False


class Trainer:
    """
    Trainer class for neural network training (LVAE or standard).
    """

    def __init__(self, model, train_loader, val_loader, device,
                 training_prm=None, data_prm=None, saving_prm=None,
                 exp_name=None, mode="train", is_lvae=False,
                 writer=None, state_dict=None):

        # Core objects
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.writer = writer
        self.state_dict = state_dict or {"epoch": 0}

        # Parameters
        self.exp_name = exp_name
        self.mode = mode
        self.is_lvae = is_lvae
        self.training_prm = training_prm or {}
        self.data_prm = data_prm or {}
        self.saving_prm = saving_prm or {}

        # Initialization
        self._init_training_prm()
        self._init_data_prm()
        self._init_saving_prm()
        self._init_logger()

        # Optimizer & scheduler
        self.optimizer, self.scheduler = misc.make_optimizer_and_scheduler(self.model, self.training_prm)


    # Init helpers
    def _init_training_prm(self):
        self.batch_size = self.training_prm.get("batch_size", 8)
        self.n_epochs = self.training_prm.get("n_epochs", 100)
        self.betaKL = self.training_prm.get("betaKL", 1.0)
        self.early_stop = self.training_prm.get("early_stop", False)
        self.pos_encod = self.training_prm.get("pos_encod", False)
        self.pbar = self.training_prm.get("pbar", False) and tqdm_available
        self.update_console = not self.pbar

        # Non-LVAE loss function
        if not self.is_lvae:
            self.loss_function = misc.get_loss_function(**self.training_prm.get("loss_function", {}))
        else:
            self.loss_function = None

    def _init_data_prm(self):
        self.volumetric = self.data_prm.get("volumetric", False)
        self.tiling_validation = False
        if self.is_lvae and self.data_prm.get("val_patch_size") is not None:
            if self.data_prm.get("val_patch_size") != self.data_prm.get("patch_size"):
                self.tiling_validation = True
                self.tiling_patch = self.data_prm.get("patch_size")
                if self.data_prm.get("downsampling", {}).get("downsamp_factor") is not None:
                    p = self.data_prm.get("downsampling").get("downsamp_factor")
                    self.tiling_patch = self.tiling_patch // p

    def _init_saving_prm(self):
        if self.saving_prm.get("saving", False):
            self.model_save_folder = self.saving_prm.get("model_save_folder")
            self.saving = True
            self.loss_path = self.model_save_folder / _loss_file_name
            self.log_path = self.model_save_folder / _log_file_name
            self.save_validation_images = self.saving_prm.get("save_validation_images", False)
        else:
            self.saving = False
            self.loss_path = None
            self.log_path = None
            self.save_validation_images = False

        if self.save_validation_images:
            self.save_validation_freq = self.saving_prm.get("save_validation_freq", 10)
            self.validation_images_folder = self.model_save_folder / "validation_images"
            os.makedirs(self.validation_images_folder, exist_ok=True)

    def _init_logger(self):
        self.logger = logging.getLogger("trainer")
        self.console_filter = EnableFilter()
        self.logfile_filter = EnableFilter()

        # Console
        console_handler = CustomStreamHandler()
        console_handler.addFilter(self.console_filter)
        self.logger.addHandler(console_handler)

        # File
        if self.saving:
            file_handler = logging.FileHandler(self.log_path, mode="a")
            formatter = logging.Formatter('%(asctime)-5s %(message)s', "%Y-%m-%d %H:%M:%S")
            file_handler.setFormatter(formatter)
            file_handler.addFilter(self.logfile_filter)
            self.logger.addHandler(file_handler)

    # Factory method for YAML config
    @classmethod
    def from_config(cls, cfg, device):
        from lisai.lib.utils import get_model, get_paths

        exp_cfg = cfg.get("experiment", {})
        exp_name = exp_cfg.get("exp_name", "default_exp")
        mode = exp_cfg.get("mode", "train")
        is_lvae = exp_cfg.get("is_lvae", False)

        data_prm = cfg.get("data", {})
        norm_prm = cfg.get("normalization", {}).get("norm_prm")
        local = data_prm.get("local", True)
        data_dir = get_paths.get_dataset_path(local=local, **data_prm)

        train_loader, val_loader, model_norm_prm, patch_info = misc.make_training_loaders(
            data_dir=data_dir,
            norm_prm=norm_prm,
            **data_prm
        )
        data_prm["patch_info"] = patch_info
        cfg["data"] = data_prm
        cfg["model_norm_prm"] = model_norm_prm

        # Noise model
        if is_lvae:
            noise_model, noise_norm = get_model.getNoiseModel(local, device, cfg.get("noise_model"))
            if cfg.get("normalization", {}).get("load_from_noise_model", False):
                cfg["normalization"]["norm_prm"] = noise_norm
        else:
            noise_model = None

        # Model
        model, state_dict = get_model.get_model_for_training(cfg, device, model_norm_prm, noise_model)

        # Saving & tensorboard
        saving_prm = misc.handle_saving(cfg)
        cfg["saving_prm"] = saving_prm
        writer = misc.handle_tensorboard(cfg)

        training_prm = cfg.get("training", {})

        return cls(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            training_prm=training_prm,
            data_prm=data_prm,
            saving_prm=saving_prm,
            exp_name=exp_name,
            mode=mode,
            is_lvae=is_lvae,
            writer=writer,
            state_dict=state_dict
        )


    # Main training loop

    def train(self):
        start_epoch = self.state_dict.get("epoch", 0)
        best_loss = self.state_dict.get("best_loss", float('inf'))

        iter_epoch = range(start_epoch, self.n_epochs)
        if self.pbar:
            iter_epoch = tqdm(iter_epoch, position=1, total=self.n_epochs, initial=start_epoch)
            iter_epoch.set_description('Epochs')

        self.initialize_log_file()
        self.logger.info('Starting Training...')

        for epoch in iter_epoch:
            try:
                save_val_imgs = self.save_validation_images and (epoch % self.save_validation_freq == 0)
                start = time.time()
                train_loss, train_kl_loss, train_recons_loss = self.train_epoch(epoch)
                val_loss, val_kl_loss, val_recons_loss = self.validate(epoch, save_imgs=save_val_imgs)
                end = time.time()

                # Update state_dict
                self.state_dict.update({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict()
                })

                if self.is_lvae:
                    self.state_dict.update({
                        "train_kl_loss": train_kl_loss,
                        "val_kl_loss": val_kl_loss,
                        "train_recons_loss": train_recons_loss,
                        "val_recons_loss": val_recons_loss,
                    })

                if self.scheduler is not None:
                    self.state_dict["scheduler"] = self.scheduler.state_dict()

                # Check for best model
                is_best = val_loss < best_loss
                if is_best:
                    best_loss = val_loss
                    self.logger.info(f"epoch {epoch}: Best model saved with best_val = {best_loss}.")

                # Scheduler step
                if self.scheduler is not None:
                    if self.training_prm.get("scheduler") == "ReduceLROnPlateau":
                        self.scheduler.step(val_loss)
                    else:
                        self.scheduler.step()

                # Save models
                if self.saving:
                    self.handle_saving(best_loss, is_best)
                    self.update_loss()

                # Tensorboard logging
                if self.writer is not None:
                    self.writer.add_scalar('Train_loss', train_loss, epoch)
                    self.writer.add_scalar('Val_loss', val_loss, epoch)

                if self.early_stop and epoch > 2:
                    self.logger.info("Early stopping.")
                    break

            except KeyboardInterrupt:
                self.handle_keyboard_interrupt(epoch, best_loss, train_loss, val_loss)
                return
            except Exception as e:
                self.handle_exception(epoch, e)

        self.finalize_logging(epoch, best_loss, val_loss)

    # -----------------------------
    # Train one epoch
    # -----------------------------
    def train_epoch(self, epoch):
        self.model.train()
        self.model.to(self.device)
        train_loss = []
        train_kl_loss = [] if self.is_lvae else None
        train_recons_loss = [] if self.is_lvae else None

        iter_loader = self.train_loader
        if self.pbar:
            iter_loader = tqdm(iter_loader, leave=False, position=0)
            iter_loader.set_description(f'Training - Epoch {epoch}')

        for batch_id, batch in enumerate(iter_loader):
            virtual_batches = tu.create_virtual_batches(batch, self.batch_size)

            for vb in zip(*virtual_batches):
                if self.is_lvae:
                    loss, kl, recons, pred = tu.compute_loss(
                        vb, self.model, self.device,
                        is_lvae=True, betaKL=self.betaKL,
                        pos_encod=self.pos_encod, volumetric=self.volumetric
                    )
                    train_kl_loss.append(kl)
                    train_recons_loss.append(recons)
                else:
                    x, y, *samp_pos = vb
                    pred = self.model(x, samp_pos)
                    loss = self.loss_function(pred, y)

                loss.backward()
                train_loss.append(loss.item())
                self.optimizer.step()
                self.optimizer.zero_grad()

                if self.update_console:
                    self.logger.info(f"Epoch {epoch}/{self.n_epochs}, Batch {batch_id}/{len(iter_loader)}")

                if self.early_stop:
                    break

        train_loss = np.mean(train_loss)
        train_kl_loss = np.mean(train_kl_loss) if self.is_lvae else None
        train_recons_loss = np.mean(train_recons_loss) if self.is_lvae else None

        return train_loss, train_kl_loss, train_recons_loss

    # -----------------------------
    # Validation
    # -----------------------------
    def validate(self, epoch, save_imgs=False):
        self.model.eval()
        self.model.to(self.device)
        val_loss = []
        val_kl_loss = [] if self.is_lvae else None
        val_recons_loss = [] if self.is_lvae else None
        val_imgs = [] if save_imgs else None

        with torch.no_grad():
            iter_val = self.val_loader
            if self.pbar:
                iter_val = tqdm(iter_val, leave=False, position=0)
                iter_val.set_description(f'Validation - Epoch {epoch}')

            for batch_id, batch in enumerate(iter_val):
                virtual_batches = tu.create_virtual_batches(batch, self.batch_size)

                for vb in zip(*virtual_batches):
                    if self.is_lvae:
                        if self.tiling_validation:
                            x, *_ = vb
                            outputs = lvae_forward_pass_tiling(x, None, self.device,
                                                              self.model,
                                                              patch_size=self.tiling_patch)
                            pred = outputs['out_mean']
                            kl = outputs['kl_loss']
                            recons = outputs['recons_loss']
                            loss = recons + self.betaKL * kl
                        else:
                            loss, kl, recons, pred = tu.compute_loss(
                                vb, self.model, self.device,
                                is_lvae=True, betaKL=self.betaKL,
                                pos_encod=self.pos_encod, volumetric=self.volumetric
                            )
                        val_kl_loss.append(kl)
                        val_recons_loss.append(recons)
                        val_loss.append(loss.item())
                    else:
                        x, y, *samp_pos = vb
                        pred = self.model(x, samp_pos)
                        val_loss.append(self.loss_function(pred, y).item())

                    if self.writer is not None:
                        tu.log_images_to_tensorboard(self.writer, x, y, pred, epoch, volumetric=self.volumetric)

                    if save_imgs:
                        val_imgs.append((x, y, pred))

        if save_imgs:
            tu.save_validation_images(val_imgs, self.validation_images_folder, volumetric=self.volumetric)

        val_loss = np.mean(val_loss)
        val_kl_loss = np.mean(val_kl_loss) if self.is_lvae else None
        val_recons_loss = np.mean(val_recons_loss) if self.is_lvae else None

        return val_loss, val_kl_loss, val_recons_loss

    # -----------------------------
    # Saving
    # -----------------------------
    def handle_saving(self, best_loss, is_best):
        epoch = self.state_dict["epoch"]
        name_best = "model_best" if self.saving_prm.get("overwrite_best", True) else f"model_epoch_{epoch}"
        folder = self.model_save_folder

        if self.saving_prm.get("state_dict", True):
            if is_best:
                torch.save(self.state_dict, folder / f"{name_best}_state_dict.pt")
            torch.save(self.state_dict, folder / "model_last_state_dict.pt")

        if self.saving_prm.get("entire_model", True):
            if is_best:
                torch.save(self.model, folder / f"{name_best}.pt")
            torch.save(self.model, folder / "model_last.pt")

    # -----------------------------
    # Logging losses
    # -----------------------------
    def update_loss(self):
        if not os.path.exists(self.loss_path):
            with open(self.loss_path, 'w') as f:
                if self.is_lvae:
                    f.write(f"{'Epoch':<10} {'Train_loss':<20} {'Val_loss':<20} {'Recons_Loss':<20} {'KL_Loss':<20}\n")
                else:
                    f.write(f"{'Epoch':<10} {'Train_loss':<20} {'Val_loss':<20}\n")

        epoch = self.state_dict["epoch"]
        train_loss = self.state_dict["train_loss"]
        val_loss = self.state_dict["val_loss"]

        with open(self.loss_path, "a") as f:
            if self.is_lvae:
                recons = self.state_dict["train_recons_loss"]
                kl = self.state_dict["train_kl_loss"]
                f.write(f"{epoch:<10} {train_loss:<20} {val_loss:<20} {recons:<20} {kl:<20}\n")
            else:
                f.write(f"{epoch:<10} {train_loss:<20} {val_loss:<20}\n")

    # -----------------------------
    # Logging helpers
    # -----------------------------
    def initialize_log_file(self):
        self.logger.info(f"Experiment: {self.exp_name}, Mode: {self.mode}, Device: {self.device}")

    def handle_keyboard_interrupt(self, epoch, best_loss, train_loss, val_loss):
        self.logger.info(f"Training manually stopped at epoch {epoch}. Best_val={best_loss}, train={train_loss}, val={val_loss}")

    def handle_exception(self, epoch, e):
        self.logger.error(f"Training stopped during epoch {epoch} due to {type(e)}: {e}")
        raise

    def finalize_logging(self, epoch, best_loss, val_loss):
        self.logger.info(f"Finished training: {epoch+1}/{self.n_epochs}. Best_val={best_loss}, last_val={val_loss}")
