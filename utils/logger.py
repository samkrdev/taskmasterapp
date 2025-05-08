# Directory: utils/logger.py
"""
Logging configuration for the application.
"""
import logging
import sys
from typing import Optional


def setup_logger(name: str = "scheduler", level: int = logging.INFO) -> logging.Logger:
    """Set up and configure a logger."""
    logger = logging.getLogger(name)

    # Avoid duplicate handlers if logger already exists
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # Create handlers
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    # Create formatters
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    # Add formatters to handlers
    console_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(console_handler)

    return logger


# Default logger for the application
logger = setup_logger()
