import logging

try:
    from tqdm import tqdm
    _tqdm_available = True
except Exception:
    _tqdm_available = False


class EnableFilter(logging.Filter):
    """
    Utility class for enabling/disabling logging messages per handler.
    """

    def __init__(self, enable: bool = True):
        super().__init__()
        self.enable = enable

    def filter(self, record: logging.LogRecord) -> bool:
        return bool(self.enable)


class CustomStreamHandler(logging.StreamHandler):
    """
    StreamHandler that:
      - enforces a simple console format
      - uses tqdm.write if available, so logs don't break progress bars
    """

    def __init__(self, use_tqdm: bool = True):
        super().__init__()
        self.setFormatter(logging.Formatter("%(name)s: %(levelname)s : %(message)s"))
        self.use_tqdm = bool(use_tqdm) and _tqdm_available
    
    @property
    def tqdm_enabled(self) -> bool:
        return self.use_tqdm

    def emit(self, record: logging.LogRecord):
        if self.use_tqdm:
            try:
                msg = self.format(record)
                tqdm.write(msg)
                self.flush()
            except Exception:
                self.handleError(record)
        else:
            super().emit(record)
