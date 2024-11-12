"""
Module for logging
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


############### LOGGING ###############
class CustomLogger(logging.Logger):
    def __init__(
        self,
        name,
        level=0,
        log_filepath: str = "logs/tmp.log",
        maxBytes: int = 10240,
    ):
        """
        (10KB file size limit)
        """
        super().__init__(name, level)

        # Setup rotating file handler
        Path(log_filepath).parent.mkdir(exist_ok=True, parents=True)
        handler = RotatingFileHandler(
            log_filepath, maxBytes=maxBytes, backupCount=0
        )
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        self.addHandler(handler)
