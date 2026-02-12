"""
Structured logging configuration for MedArchive RAG.

Provides JSON-formatted logs for production observability and
human-readable logs for local development.
"""

import logging
import sys
from typing import Any, Dict

from rich.console import Console
from rich.logging import RichHandler


def setup_logging(
    service_name: str,
    log_level: str = "INFO",
    log_format: str = "text",
    log_output: str = "stdout",
) -> logging.Logger:
    """
    Configure structured logging for a service.

    Args:
        service_name: Name of the service (e.g., 'api', 'ingestion')
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: 'json' for production, 'text' for development
        log_output: 'stdout', 'file', or 'both'

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(service_name)
    logger.setLevel(getattr(logging, log_level.upper()))
    logger.handlers.clear()  # Remove existing handlers

    if log_format == "json":
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter(service_name=service_name))
        logger.addHandler(handler)
    else:
        # Rich handler for beautiful local dev logs
        console = Console(stderr=False)
        handler = RichHandler(
            console=console,
            rich_tracebacks=True,
            tracebacks_show_locals=True,
            show_time=True,
            show_path=True,
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)

    # Optionally add file handler
    if log_output in ["file", "both"]:
        file_handler = logging.FileHandler(f"logs/{service_name}.log")
        file_handler.setFormatter(JSONFormatter(service_name=service_name))
        logger.addHandler(file_handler)

    # Suppress noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    return logger


class JSONFormatter(logging.Formatter):
    """
    Custom JSON formatter for structured logging.

    Converts log records to JSON for ingestion by observability platforms
    (e.g., Azure Monitor, Datadog, Elastic).
    """

    def __init__(self, service_name: str):
        super().__init__()
        self.service_name = service_name

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON string."""
        import json
        from datetime import datetime

        log_data: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "service": self.service_name,
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields from LoggerAdapter or explicit extra kwargs
        if hasattr(record, "extra"):
            log_data.update(record.extra)

        return json.dumps(log_data)


class LoggerAdapter(logging.LoggerAdapter):
    """
    Adapter to inject contextual information into all log messages.

    Usage:
        logger = setup_logging("api")
        request_logger = LoggerAdapter(logger, {"request_id": request.id})
        request_logger.info("Processing query", extra={"user_id": user.id})
    """

    def process(self, msg: str, kwargs: Any) -> tuple:
        """Inject context into log message."""
        # Merge adapter context with per-call extra
        if "extra" not in kwargs:
            kwargs["extra"] = {}
        kwargs["extra"].update(self.extra)
        return msg, kwargs
