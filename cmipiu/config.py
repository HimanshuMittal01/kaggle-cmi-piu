"""
Module for logging
"""

import logging
from logging.handlers import RotatingFileHandler


############### LOGGING ###############
class CustomLogger(logging.Logger):
    def __init__(self, name, level=0):
        super().__init__(name, level)

        # Setup rotating file handler (10KB file size limit)
        handler = RotatingFileHandler("tmp.log", maxBytes=1024 * 10, backupCount=0)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        self.addHandler(handler)
