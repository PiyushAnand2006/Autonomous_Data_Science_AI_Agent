"""
AADS Structured Logging.

Provides a consistent logging setup using ``structlog`` for machine-readable
JSON output and ``rich`` for pretty console output during development.

Usage:
    from aads.core.logging import get_logger
    logger = get_logger(__name__)
    logger.info("profiling_complete", rows=10000, cols=42)
"""

from __future__ import annotations

import logging
import sys

import structlog


_CONFIGURED = False


def configure_logging(
    level: str = "INFO",
    json_output: bool = False,
) -> None:
    """Configure structlog + stdlib logging for the AADS process.

    Call once at application startup. Subsequent calls are no-ops.

    Args:
        level: Logging level string (DEBUG, INFO, WARNING, ERROR).
        json_output: If True, emit JSON lines. Otherwise, pretty console output.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return
    _CONFIGURED = True

    log_level = getattr(logging, level.upper(), logging.INFO)

    # stdlib root logger
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )

    # Shared processors
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if json_output:
        renderer: structlog.types.Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    # Apply formatter to all handlers on the root logger
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        handler.setFormatter(formatter)


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a bound structlog logger.

    Args:
        name: Logger name, typically ``__name__`` of the calling module.

    Returns:
        A structlog BoundLogger instance.
    """
    configure_logging()  # ensure configured (idempotent)
    return structlog.get_logger(name)
