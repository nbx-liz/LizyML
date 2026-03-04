"""Loader — restore Model artifacts from a directory.

Security note: joblib.load() executes arbitrary Python code.
Only load artifacts from trusted sources.

Raises DESERIALIZATION_FAILED when:
- metadata.json is missing or malformed
- format_version is unknown
- Required metadata fields are absent
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib

from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.persistence.exporter import FORMAT_VERSION

_REQUIRED_METADATA_KEYS = frozenset(
    {"format_version", "task", "feature_names", "config", "run_id"}
)


def load(path: str | Path) -> tuple[Any, Any, dict[str, Any]]:
    """Load Model artifacts from *path*.

    Args:
        path: Directory previously created by
            :func:`~lizyml.persistence.exporter.export`.

    Returns:
        ``(fit_result, refit_result, metadata)`` triple.

    Raises:
        LizyMLError with DESERIALIZATION_FAILED on any validation or I/O error.

    Warning:
        Only load from trusted sources — joblib uses pickle internally.
    """
    src = Path(path)
    metadata_path = src / "metadata.json"
    fit_pkl = src / "fit_result.pkl"
    refit_pkl = src / "refit_model.pkl"

    # --- Validate directory and metadata -------------------------------------
    if not src.is_dir():
        raise LizyMLError(
            code=ErrorCode.DESERIALIZATION_FAILED,
            user_message=f"Export directory not found: '{path}'",
            context={"path": str(path)},
        )

    if not metadata_path.exists():
        raise LizyMLError(
            code=ErrorCode.DESERIALIZATION_FAILED,
            user_message=f"metadata.json not found in '{path}'.",
            context={"path": str(path)},
        )

    try:
        metadata: dict[str, Any] = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise LizyMLError(
            code=ErrorCode.DESERIALIZATION_FAILED,
            user_message=f"Failed to parse metadata.json: {exc}",
            context={"path": str(path)},
            cause=exc,
        ) from exc

    # Check required fields
    missing = _REQUIRED_METADATA_KEYS - set(metadata.keys())
    if missing:
        raise LizyMLError(
            code=ErrorCode.DESERIALIZATION_FAILED,
            user_message=f"metadata.json is missing required fields: {sorted(missing)}",
            context={"missing": sorted(missing)},
        )

    # Check format_version
    fv = metadata.get("format_version")
    if fv != FORMAT_VERSION:
        raise LizyMLError(
            code=ErrorCode.DESERIALIZATION_FAILED,
            user_message=(
                f"Unsupported format_version={fv!r}. "
                f"This version of LizyML supports format_version={FORMAT_VERSION}."
            ),
            context={"format_version": fv, "supported": FORMAT_VERSION},
        )

    # --- Load pickled artifacts ----------------------------------------------
    for pkl_path in (fit_pkl, refit_pkl):
        if not pkl_path.exists():
            raise LizyMLError(
                code=ErrorCode.DESERIALIZATION_FAILED,
                user_message=f"Required artifact not found: '{pkl_path.name}'",
                context={"path": str(pkl_path)},
            )

    try:
        fit_result = joblib.load(fit_pkl)
        refit_result = joblib.load(refit_pkl)
    except Exception as exc:
        raise LizyMLError(
            code=ErrorCode.DESERIALIZATION_FAILED,
            user_message=f"Failed to load model artifacts: {exc}",
            context={"path": str(path)},
            cause=exc,
        ) from exc

    return fit_result, refit_result, metadata
