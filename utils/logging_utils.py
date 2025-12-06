"""Lightweight logging helpers."""
import logging
from typing import Optional


_DEF_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def get_logger(name: str, level: int = logging.INFO, fmt: Optional[str] = None) -> logging.Logger:
    """Create and configure a logger with stream handler."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(fmt or _DEF_FORMAT))
    logger.addHandler(handler)
    logger.setLevel(level)
    return logger


__all__ = ["get_logger"]
