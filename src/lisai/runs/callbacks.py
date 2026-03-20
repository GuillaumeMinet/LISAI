from __future__ import annotations

from pathlib import Path

from .lifecycle import update_run_progress


class RunMetadataCallback:
    def __init__(self, run_dir: str | Path, *, max_epoch: int | None, logger=None):
        self.run_dir = Path(run_dir)
        self.max_epoch = max_epoch
        self.logger = logger

    def on_epoch_end(self, trainer, epoch: int, logs: dict):
        try:
            update_run_progress(
                self.run_dir,
                last_epoch=epoch,
                max_epoch=self.max_epoch,
                val_loss=logs.get("val_loss"),
            )
        except Exception as exc:
            _log_warning(
                self.logger,
                f"Failed to update run metadata for epoch {epoch}: {type(exc).__name__}: {exc}",
            )


def _log_warning(logger, message: str):
    if logger is None:
        return

    warning = getattr(logger, "warning", None)
    if callable(warning):
        warning(message)
        return

    info = getattr(logger, "info", None)
    if callable(info):
        info(message)


__all__ = ["RunMetadataCallback"]
