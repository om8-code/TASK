# logging_config.py
"""Centralised logging configuration for the FastAPI service.

The module provides:

* :func:`configure_logging` – idempotent function that sets up the root logger
  with a JSON formatter.  It reads ``LOG_LEVEL`` and ``LOG_FILE`` from the
  environment but also accepts explicit arguments for programmatic use.
* :func:`get_logger` – thin wrapper around :pyfunc:`logging.getLogger` that
  guarantees the logging system is configured before the logger is returned.

The JSON formatter makes log aggregation trivial when the service runs in a
container orchestration platform (e.g. Kubernetes, Docker) or is shipped to a
log‑management service such as Loki, Elastic, or CloudWatch.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from typing import Any, Mapping

__all__ = ["configure_logging", "get_logger"]

# ---------------------------------------------------------------------------
# JSON formatter
# ---------------------------------------------------------------------------


class JsonFormatter(logging.Formatter):
    """Format a :class:`logging.LogRecord` as a single‑line JSON string.

    The output contains a minimal set of fields that are useful for most
    observability pipelines:

    * ``timestamp`` – ISO‑8601 UTC timestamp with millisecond precision.
    * ``level`` – Log level name (e.g. ``INFO``).
    * ``logger`` – Name of the logger that emitted the record.
    * ``message`` – The formatted log message.
    * ``pathname`` – Source file path.
    * ``lineno`` – Line number in the source file.
    * ``exception`` – Optional stringified exception information when
      ``exc_info`` is present.
    * Any extra ``record.__dict__`` items that are not part of the standard
      ``LogRecord`` attributes are merged under the ``extra`` key.
    """

    # Standard LogRecord attributes that we *do not* treat as extra data.
    _standard_attrs = {
        "name",
        "msg",
        "args",
        "levelname",
        "levelno",
        "pathname",
        "filename",
        "module",
        "exc_info",
        "exc_text",
        "stack_info",
        "lineno",
        "funcName",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "processName",
        "process",
    }

    def format(self, record: logging.LogRecord) -> str:
        # Ensure the message is rendered (handles ``%``‑style formatting).
        record.message = record.getMessage()

        log_record: Mapping[str, Any] = {
            "timestamp": datetime.utcfromtimestamp(record.created).isoformat(timespec="milliseconds") + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.message,
            "pathname": record.pathname,
            "lineno": record.lineno,
        }

        # Attach exception information if present.
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)

        # Capture any user‑supplied ``extra`` fields.
        extra = {k: v for k, v in record.__dict__.items() if k not in self._standard_attrs}
        if extra:
            log_record["extra"] = extra

        # ``json.dumps`` guarantees ASCII output and a single line.
        return json.dumps(log_record, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Configuration helper
# ---------------------------------------------------------------------------


def _create_handler(stream: Any = sys.stdout) -> logging.Handler:
    """Create a :class:`logging.StreamHandler` with the JSON formatter.

    Parameters
    ----------
    stream:
        The output stream – defaults to ``sys.stdout``.  Supplying a custom
        stream is useful for unit‑tests.
    """
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    return handler


def configure_logging(log_level: str | None = None, log_file: str | None = None) -> None:
    """Configure the root logger for the application.

    The function is safe to call multiple times – duplicate handlers are not
    added.  Configuration order:

    1. Environment variables ``LOG_LEVEL`` and ``LOG_FILE`` are consulted.
    2. Explicit arguments ``log_level`` and ``log_file`` override the env
       values.
    3. If ``log_file`` is provided, a rotating file handler (5 MiB per file,
       3 backups) is attached in addition to the standard stream handler.
    """
    # Resolve configuration values.
    env_level = os.getenv("LOG_LEVEL")
    env_file = os.getenv("LOG_FILE")
    level = log_level or env_level or "INFO"
    file_path = log_file or env_file

    # Normalise the level name – ``logging`` raises ``ValueError`` for unknown
    # levels, which we surface as a clear configuration error.
    try:
        numeric_level = logging.getLevelName(level.upper())
        if isinstance(numeric_level, str):  # getLevelName returns the name back if unknown
            raise ValueError
    except Exception as exc:
        raise ValueError(f"Invalid LOG_LEVEL '{level}'. Must be a valid logging level name.") from exc

    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Prevent handler duplication – we keep a set of handler types we already
    # attached.
    existing_handler_classes = {type(h) for h in root_logger.handlers}

    # Always ensure a stream handler is present.
    if logging.StreamHandler not in existing_handler_classes:
        root_logger.addHandler(_create_handler())

    # Optional file handler.
    if file_path and logging.FileHandler not in existing_handler_classes:
        from logging.handlers import RotatingFileHandler

        file_handler = RotatingFileHandler(
            filename=file_path,
            maxBytes=5 * 1024 * 1024,  # 5 MiB
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setFormatter(JsonFormatter())
        root_logger.addHandler(file_handler)

    # Silence the ``uvicorn.error`` logger if it exists – FastAPI already emits
    # its own structured logs via this configuration.
    uvicorn_error_logger = logging.getLogger("uvicorn.error")
    uvicorn_error_logger.handlers = root_logger.handlers
    uvicorn_error_logger.setLevel(root_logger.level)


def get_logger(name: str = __name__) -> logging.Logger:
    """Return a logger instance, ensuring the logging system is configured.

    The first call to :func:`get_logger` triggers :func:`configure_logging`
    using the default environment‑driven configuration.  Subsequent calls are
    cheap because the configuration function is idempotent.
    """
    # Lazy configuration – only configure once the first logger is requested.
    if not logging.getLogger().handlers:
        configure_logging()
    return logging.getLogger(name)

