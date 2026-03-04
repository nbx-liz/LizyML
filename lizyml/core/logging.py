"""Structured logging utilities for LizyML."""

from __future__ import annotations

import logging
import uuid
from typing import Any


def generate_run_id() -> str:
    """Generate a unique run identifier."""
    return str(uuid.uuid4())


def get_logger(name: str) -> logging.Logger:
    """Return a logger namespaced under 'lizyml'.

    All loggers share the 'lizyml' root so users can configure them
    with a single ``logging.getLogger('lizyml').setLevel(...)`` call.
    """
    return logging.getLogger(f"lizyml.{name}")


def log_event(
    logger: logging.Logger,
    event: str,
    level: int = logging.INFO,
    **fields: Any,
) -> None:
    """Emit a structured log record with key=value fields.

    Args:
        logger: Target logger.
        event: Short event label (e.g. ``"fit.start"``).
        level: Logging level (default INFO).
        **fields: Arbitrary key-value context (run_id, fold, config_hash, …).
                  Do NOT include raw PII or large data payloads.
    """
    parts = [f"event={event!r}"]
    parts.extend(f"{k}={v!r}" for k, v in fields.items())
    logger.log(level, " ".join(parts))
