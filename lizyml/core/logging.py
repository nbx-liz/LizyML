"""Structured logging utilities for LizyML."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
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


def setup_output_dir(output_dir: str | Path, run_id: str) -> Path:
    """Create run-specific output directory and configure file logging.

    Creates ``{output_dir}/{run_id}/`` and adds a :class:`FileHandler`
    to the ``'lizyml'`` root logger writing to ``run.log``.

    Args:
        output_dir: Base output directory.
        run_id: Unique run identifier.

    Returns:
        Path to the created run directory.
    """
    run_dir = Path(output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    log_path = run_dir / "run.log"
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
    )
    root = logging.getLogger("lizyml")
    root.addHandler(handler)
    if root.level == logging.NOTSET:
        root.setLevel(logging.INFO)

    return run_dir
