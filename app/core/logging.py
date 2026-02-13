"""Centralized logging configuration with JSON-formatted extras."""

import json
import logging
import sys


class JSONExtrasFormatter(logging.Formatter):
    """Formatter that outputs a readable log line with extras as JSON.

    Output format:
        2024-01-15 10:30:45 | INFO | app.module | Message {"key": "value"}
    """

    RESERVED_ATTRS = frozenset(
        {
            "args",
            "asctime",
            "created",
            "exc_info",
            "exc_text",
            "filename",
            "funcName",
            "levelname",
            "levelno",
            "lineno",
            "message",
            "module",
            "msecs",
            "msg",
            "name",
            "pathname",
            "process",
            "processName",
            "relativeCreated",
            "stack_info",
            "taskName",
            "thread",
            "threadName",
        }
    )

    def format(self, record: logging.LogRecord) -> str:
        # Build the base message
        record.message = record.getMessage()
        timestamp = self.formatTime(record, self.datefmt)
        base = f"{timestamp} | {record.levelname:<8} | {record.name} | {record.message}"

        # Collect extras (anything not in the reserved set)
        extras = {
            k: v
            for k, v in record.__dict__.items()
            if k not in self.RESERVED_ATTRS and not k.startswith("_")
        }

        if extras:
            try:
                extras_str = json.dumps(extras, default=str, ensure_ascii=False)
                base = f"{base} {extras_str}"
            except (TypeError, ValueError):
                pass

        # Append exception info if present
        if record.exc_info and not record.exc_text:
            record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            base = f"{base}\n{record.exc_text}"

        return base


def setup_logging() -> None:
    """Configure the root 'app' logger with console output and JSON extras."""
    logger = logging.getLogger("app")
    logger.setLevel(logging.INFO)

    # Avoid adding duplicate handlers if called multiple times
    if logger.handlers:
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    handler.setFormatter(
        JSONExtrasFormatter(datefmt="%Y-%m-%d %H:%M:%S")
    )

    logger.addHandler(handler)

    # Prevent propagation to root logger (avoid duplicate output)
    logger.propagate = False
