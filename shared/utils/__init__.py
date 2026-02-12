"""Shared utilities package."""

import logging as _logging

from shared.utils.config import Settings, get_settings
from shared.utils.logging import LoggerAdapter, setup_logging


def get_logger(name: str) -> _logging.Logger:
    """
    Get a logger instance for the given name.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance
    """
    return _logging.getLogger(name)


__all__ = ["get_settings", "Settings", "setup_logging", "LoggerAdapter", "get_logger"]
