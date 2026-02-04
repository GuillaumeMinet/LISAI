import logging

try:
    from tqdm import tqdm
    tqdm_available = True
except:
    tqdm_available = False

    
class EnableFilter(logging.Filter):
    """
    Utility class for enabling/desabling logging messages in console.
    """
    
    def __init__(self):
        super().__init__()
        self.enable = True

    def filter(self, record):
        return self.enable
    
    
class CustomStreamHandler(logging.StreamHandler):
    """
    Subclass of StreamHandler to:
      - enforce the format of all the consoles output.
      - use tdqm.write in order to use tdqm progress bar
    """
    def __init__(self,use_tqdm = True):
        super().__init__()
        self.setFormatter(logging.Formatter('%(name)s: %(levelname)s : %(message)s'))
        self.use_tqdm = use_tqdm and tqdm_available
    
    def emit(self, record):
        if self.use_tqdm:
            try:
                msg = self.format(record)
                tqdm.write(msg)
                self.flush()
            except Exception:
                self.handleError(record)
        else:
            super().emit(record)