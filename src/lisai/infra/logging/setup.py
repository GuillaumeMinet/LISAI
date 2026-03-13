# src/lisai/infra/logging/setup.py

from __future__ import annotations

import logging
import sys
from pathlib import Path

from .handlers import CustomStreamHandler, EnableFilter


def setup_logger(
    *,
    name: str = "lisai",
    level: int = logging.INFO,
    log_file: Path | None = None,
    use_tqdm: bool = True,
):
    """
    Creates/overwrites a logger with:
      - tqdm-friendly console handler
      - optional file handler
      - per-handler EnableFilter for selective output control
    Returns:
      logger, console_filter, file_filter
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # Clear existing handlers to avoid duplicates in notebooks/re-runs
    for h in list(logger.handlers):
        logger.removeHandler(h)

    console_filter = EnableFilter(True)
    file_filter = EnableFilter(True)

    # Console handler
    ch = CustomStreamHandler(use_tqdm=use_tqdm)
    ch.addFilter(console_filter)
    ch.setLevel(level)

    # Ensure stream is stdout (CustomStreamHandler default stream is sys.stderr)
    try:
        ch.stream = sys.stdout
    except Exception:
        pass

    logger.addHandler(ch)

    if use_tqdm and not ch.tqdm_enabled:
        logger.warning("training.pbar=True but tqdm is not installed.")

    # File handler
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        fh = logging.FileHandler(str(log_file))
        fh.setFormatter(logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s"))
        fh.addFilter(file_filter)
        fh.setLevel(level)
        logger.addHandler(fh)
    else:
        file_filter.enable = False  # if no file handler, keep semantics consistent

    return logger, console_filter, file_filter
