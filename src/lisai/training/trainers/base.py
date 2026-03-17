import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch

from lisai.config.models import ResolvedExperiment
from lisai.training.checkpointing import CheckpointManager

try:
    from tqdm import tqdm
    _tqdm_available = True
except Exception:
    _tqdm_available = False


TrainingStopReason = Literal["completed", "early_stopped", "interrupted", "no_epochs"]


@dataclass(frozen=True)
class TrainingOutcome:
    reason: TrainingStopReason
    last_completed_epoch: int | None


class BaseTrainer(ABC):
    def __init__(
        self,
        *,
        model,
        train_loader,
        val_loader,
        device,
        cfg: ResolvedExperiment,
        run_dir,
        volumetric=False,
        writer=None,
        state_dict=None,
        callbacks=None,
        patch_info=None,
        console_filter=None,
        file_filter=None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        self.cfg = cfg
        self.mode = cfg.experiment.mode
        self.volumetric = volumetric

        self.training_prm = cfg.training.model_dump()
        self.data_cfg = cfg.data
        self.saving_prm = cfg.saving.model_dump()
        self.tensorboard_prm = cfg.tensorboard.model_dump()

        self.run_dir = Path(run_dir) if run_dir is not None else None
        self.exp_name = self.run_dir.name if self.run_dir is not None else cfg.experiment.exp_name
        self.writer = writer

        self.callbacks = callbacks or []
        self.patch_info = patch_info or getattr(self.data_cfg, "patch_info", None)
        self.console_filter = console_filter
        self.file_filter = file_filter
        self._training_log_initialized = False
        
        # training parameters
        self.batch_size = self.training_prm.get("batch_size")
        self.n_epochs = self.training_prm.get("n_epochs")
        self.early_stop = self.training_prm.get("early_stop", False)
        self.pos_encod = self.training_prm.get("pos_encod", False)

        # progress bar policy
        self.pbar = bool(self.training_prm.get("progress_bar")) and _tqdm_available
        self.update_console = not self.pbar

        # state dict
        self.state_dict = state_dict if state_dict is not None else {"epoch": -1}

        # optimizer + scheduler
        self.optimizer, self.scheduler = self._make_optimizer_and_scheduler()

        # checkpoint manager (file outputs)
        self.ckpt = CheckpointManager(
            self.run_dir,
            self.saving_prm,
            is_lvae=self.is_lvae,
        )

        # logger (system sets handlers)
        self.logger = logging.getLogger("lisai")

    # PROPERTY SUBCLASSES MUST DEFINE #
    @property
    @abstractmethod
    def is_lvae(self) -> bool:
        ...

    # shared batch utilities 
    def _split_batch(self, batch, *, warn_once: bool = False):
        if int(batch[0].shape[0] // self.batch_size) == 0:
            virtual_batches = [(tensor,) for tensor in batch]
            if warn_once:
                self.logger.warning("batch_size bigger than # of virtual batches, not optimal")
            return virtual_batches
        return [torch.split(tensor, self.batch_size, dim=0) for tensor in batch]

    def _prepare_batch(self, x, y, samp_pos):
        if torch.isnan(y).all().item():
            y = None
        else:
            y = y.to(self.device)

        if self.pos_encod:
            assert samp_pos is not None
            samp_pos = samp_pos.to(self.device)
        else:
            samp_pos = None

        if len(x.shape) == 4 and self.volumetric:
            x = x.unsqueeze(1)
            if y is not None:
                y = y.unsqueeze(1)

        x = x.to(self.device, dtype=torch.float)
        return x, y, samp_pos
    

    
    # OPTIMIZER AND SCHEDULER 
    def _make_optimizer_and_scheduler(self):
        """
        Creates optimizer + scheduler from training_prm.

        Expected config patterns (flexible):
        training:
            optimizer: "Adam" | {"name": "Adam", "lr": 1e-4, ...}
            scheduler: null | "StepLR" | "ReduceLROnPlateau" | {"name": "...", ...}
        """

        prm = self.training_prm or {}

        # extract optimizer name and optional arguments
        opt_cfg = prm.get("optimizer", "Adam")
        if isinstance(opt_cfg, str):
            # if no optimizer arguments given
            opt_name = opt_cfg
            opt_kwargs = {}
        elif isinstance(opt_cfg, dict):
            # separate optional arguments from optimizer name 
            opt_name = opt_cfg.get("name", "Adam")
            opt_kwargs = {k: v for k, v in opt_cfg.items() if k != "name"}
        else:
            raise ValueError(f"Invalid training.optimizer type: {type(opt_cfg)}")
        

        # resolve learning_rate, add it to opt_kwargs
        lr = opt_kwargs.get("lr", None)
        if lr is None:
            lr = prm.get("learning_rate", None)
        if lr is None:
            lr = prm.get("lr", None)
        if lr is None:
            lr = 1e-4
        opt_kwargs["lr"] = lr

        # init optimizer
        if not hasattr(torch.optim, opt_name):
            raise ValueError(f"Unknown optimizer '{opt_name}'. Available in torch.optim: {dir(torch.optim)}")
        OptimCls = getattr(torch.optim, opt_name)
        optimizer = OptimCls(self.model.parameters(), **opt_kwargs)

        # restore optimizer state if provided (continue training)
        if isinstance(self.state_dict, dict) and "optimizer_state_dict" in self.state_dict:
            try:
                optimizer.load_state_dict(self.state_dict["optimizer_state_dict"])
            except Exception as e:
                self.logger.warning(f"Failed to load optimizer_state_dict: {type(e)} {e}")

        # Scheduler
        sch_cfg = prm.get("scheduler", None)
        scheduler = None
        self._scheduler_name = None  # used in train loop to decide how to step

        if sch_cfg is None or sch_cfg is False:
            return optimizer, None

        if isinstance(sch_cfg, str):
            sch_name = sch_cfg
            sch_kwargs = {}
        elif isinstance(sch_cfg, dict):
            sch_name = sch_cfg.get("name")
            if not sch_name:
                raise ValueError("training.scheduler dict must contain a 'name' key")
            sch_kwargs = {k: v for k, v in sch_cfg.items() if k != "name"}
        else:
            raise ValueError(f"Invalid training.scheduler type: {type(sch_cfg)}")

        if not hasattr(torch.optim.lr_scheduler, sch_name):
            raise ValueError(
                f"Unknown scheduler '{sch_name}'. Available in torch.optim.lr_scheduler: {dir(torch.optim.lr_scheduler)}"
            )

        SchedCls = getattr(torch.optim.lr_scheduler, sch_name)
        scheduler = SchedCls(optimizer, **sch_kwargs)
        self._scheduler_name = sch_name

        # restore scheduler state if provided
        if isinstance(self.state_dict, dict):
            sch_state = self.state_dict.get("scheduler_state_dict") or self.state_dict.get("scheduler")
            if sch_state is not None:
                try:
                    scheduler.load_state_dict(sch_state)
                except Exception as e:
                    self.logger.warning(f"Failed to load scheduler state: {type(e)} {e}")

        return optimizer, scheduler



    # =================== #
    #    TRAINING LOOP    #
    # =================== #
    def train(self):
        last_completed_epoch = int(self.state_dict.get("epoch", -1))
        start_epoch = max(last_completed_epoch + 1, 0)
        self._initialize_log_file()
        
        if start_epoch >= self.n_epochs:
            self.logger.info(
                f"No epochs to run: start_epoch={start_epoch}, n_epochs={self.n_epochs}"
            )
            return TrainingOutcome(
                reason="no_epochs",
                last_completed_epoch=last_completed_epoch if last_completed_epoch >= 0 else None,
            )

        best_loss = self.state_dict.get("best_loss", float("inf"))
        last_train_loss = self.state_dict.get("train_loss")
        last_val_loss = self.state_dict.get("val_loss")
        iter_epoch = range(start_epoch, self.n_epochs)
        if self.pbar:
            iter_epoch = tqdm(iter_epoch, position=1, total=self.n_epochs, initial=start_epoch)
            iter_epoch.set_description("Epochs")

        self.logger.info("Starting Training...")
        stop_reason: TrainingStopReason = "completed"

        for epoch in iter_epoch:
            try:
                train_loss = None
                val_loss = None
                save_val_imgs = bool(self.saving_prm.get("validation_images", False)) and (
                    epoch % int(self.saving_prm.get("validation_freq", 10)) == 0
                )

                train_metrics = self.train_epoch(epoch)
                val_metrics = self.validate(epoch, save_imgs=save_val_imgs)

                train_loss = train_metrics["loss"]
                val_loss = val_metrics["loss"]
                last_train_loss = train_loss
                last_val_loss = val_loss

                self.state_dict.update(
                    {
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                    }
                )

                if self.is_lvae:
                    self.state_dict.update(
                        {
                            "train_kl_loss": train_metrics["kl_loss"],
                            "val_kl_loss": val_metrics["kl_loss"],
                            "train_recons_loss": train_metrics["recons_loss"],
                            "val_recons_loss": val_metrics["recons_loss"],
                        }
                    )

                if self.scheduler is not None:
                    self.state_dict["scheduler"] = self.scheduler.state_dict()

                is_best = val_loss < best_loss
                if is_best:
                    best_loss = val_loss
                    self.logger.info(f"epoch {epoch}: Best model saved with best_val = {best_loss}.")

                if self.scheduler is not None:
                    if getattr(self, "_scheduler_name", None) == "ReduceLROnPlateau":
                        self.scheduler.step(val_loss)
                    else:
                        self.scheduler.step()

                    # keep state_dict in sync for continue_training
                    self.state_dict["scheduler_state_dict"] = self.scheduler.state_dict()

                if self.saving_prm.get("enabled", True) and self.run_dir is not None:
                    self.ckpt.save(state_dict=self.state_dict, model=self.model, best_loss=best_loss, is_best=is_best)
                    self.ckpt.update_loss_file(epoch, train_metrics, val_metrics)

                # callbacks at epoch end
                logs = {"train_loss": train_loss, "val_loss": val_loss}
                for cb in self.callbacks:
                    cb.on_epoch_end(self, epoch, logs)
                last_completed_epoch = epoch

                if self.early_stop and epoch > 2:
                    self.logger.info("Early stopping.")
                    stop_reason = "early_stopped"
                    break

            except KeyboardInterrupt:
                self._log_keyboard_interrupt(epoch, best_loss, last_train_loss, last_val_loss)
                return TrainingOutcome(
                    reason="interrupted",
                    last_completed_epoch=last_completed_epoch if last_completed_epoch >= 0 else None,
                )
            except Exception as e:
                self.logger.error(
                    f"Training stopped during epoch {epoch}, because of error:\n{type(e)}\n{e}\n"
                )
                raise

        self._log_training_finished(epoch, best_loss, last_val_loss)
        return TrainingOutcome(
            reason=stop_reason,
            last_completed_epoch=last_completed_epoch if last_completed_epoch >= 0 else None,
        )

    
    # update console helper
    def _update_console_new_batch(self,epoch,batch_id,total_batches):
        """ Update console with epoch and batch number, without logging into log file."""
        if self.file_filter is not None:
            self.file_filter.enable = False
        self.logger.info(f"epochs: {epoch}/{self.n_epochs}, batch_id: {batch_id}/{total_batches}")
        if self.file_filter is not None:
            self.file_filter.enable = True

    def _initialize_log_file(self):
        if self._training_log_initialized:
            return

        self._training_log_initialized = True
        if self.file_filter is not None:
            self.file_filter.enable = True

        header = self._build_training_log_header()
        if not header:
            return

        console_enabled = None
        if self.console_filter is not None:
            console_enabled = self.console_filter.enable
            self.console_filter.enable = False

        try:
            self.logger.info(header)
        finally:
            if self.console_filter is not None and console_enabled is not None:
                self.console_filter.enable = console_enabled

    def _build_training_log_header(self) -> str:
        if self.run_dir is None:
            return ""

        computer = os.environ.get("COMPUTERNAME", "Unknown")
        gpu = self._device_name()
        patch_txt = self._patch_info_text()

        if self.mode == "train":
            return (
                f"\nExperiment name: {self.exp_name}\n"
                f"Computer: {computer}\n"
                f"Running on: {gpu}\n\n"
                f"{patch_txt}"
            )

        if self.mode == "retrain":
            return (
                f"\nExperiment name: {self.exp_name}\n"
                f"Computer: {computer}\n"
                f"Running on: {gpu}\n\n"
                "Retrain mode - starting from previously trained model.\n"
                "Check cfg file and 'retrain_origin_model' folder for details. \n\n"
                f"{patch_txt}"
            )

        if self.mode == "continue_training":
            return f"Continue training  mode. Computer: {computer}, with {gpu}.\n"

        return ""

    def _patch_info_text(self) -> str:
        if not self.patch_info:
            return ""

        train_patch = self.patch_info.get("train_patch")
        val_patch = self.patch_info.get("val_patch")
        if train_patch is None or val_patch is None:
            return ""

        return (
            f"Training patches: {train_patch}.\n"
            f"Validation patches {val_patch}.\n\n"
        )

    def _device_name(self) -> str:
        if getattr(self.device, "type", None) != "cuda":
            return "CPU"

        try:
            device_index = self.device.index if self.device.index is not None else torch.cuda.current_device()
            return torch.cuda.get_device_name(device_index)
        except Exception:
            return "CPU"

    def _log_keyboard_interrupt(self, epoch: int, best_loss: float, train_loss, val_loss):
        if train_loss is None or val_loss is None:
            self.logger.info(f"Training manually stopped during epoch {epoch}.")
            return

        self.logger.info(
            f"Training manually stopped during epoch {epoch}.\n"
            f"Model perf: best_val_loss: {best_loss} - "
            f"current_train_loss: {train_loss} - "
            f"current_val_loss: {val_loss}.\n"
        )

    def _log_training_finished(self, epoch: int, best_loss: float, val_loss):
        if val_loss is None:
            self.logger.info(f"Finished training: {epoch+1}/{self.n_epochs} epochs.")
            return

        self.logger.info(
            f"Finished training: {epoch+1}/{self.n_epochs} epochs.\n"
            f"Model perf: best_val_loss: {best_loss} - "
            f"current_val_loss: {val_loss}.\n"
        )

    # must be implemented in children
    @abstractmethod
    def train_epoch(self, epoch: int) -> dict:
        ...

    @abstractmethod
    def validate(self, epoch: int, save_imgs: bool = False) -> dict:
        ...
