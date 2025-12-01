import logging
import os
from typing import Optional

def get_logger(name: str = "app", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        logger.setLevel(level)
        fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

        # Log ra màn hình console
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(console_handler)

        logger.propagate = False
    else:
        logger.setLevel(level)
        
    return logger