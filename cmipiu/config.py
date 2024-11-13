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

        # Setup stream handler and rotating file handler
        Path(log_filepath).parent.mkdir(exist_ok=True, parents=True)

        stream_handler = logging.StreamHandler()
        file_handler = RotatingFileHandler(
            log_filepath, maxBytes=maxBytes, backupCount=0
        )

        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)

        self.addHandler(stream_handler)
        self.addHandler(file_handler)
