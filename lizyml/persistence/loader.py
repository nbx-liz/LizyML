"""Loader — restore Model artifacts from a directory.

Security note: joblib.load() executes arbitrary Python code.
Only load artifacts from trusted sources.

Raises DESERIALIZATION_FAILED when:
- metadata.json is missing or malformed
- format_version is unknown
- Required metadata fields are absent
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import joblib

from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.core.types.fit_result import FitResult
from lizyml.core.types.target_encoder import TargetEncoder
from lizyml.persistence.exporter import FORMAT_VERSION

_REQUIRED_METADATA_KEYS = frozenset(
    {"format_version", "task", "feature_names", "config", "run_id"}
)

# Format versions this loader can read. Older versions are upgraded in-memory
# via _migrate_fit_result so behavior remains equivalent to a fresh fit on
# the current library (numeric targets).
_SUPPORTED_FORMAT_VERSIONS = frozenset({1, 2})


def _migrate_fit_result(fit_result: Any, source_version: int) -> Any:
    """Upgrade a deserialized FitResult to the current FORMAT_VERSION.

    H-0070 (v1 → v2): inject a no-op :class:`TargetEncoder` so consumers
    can call ``inverse_transform`` unconditionally. v1 artifacts only ever
    held numeric targets, so a no-op encoder yields identical behavior.
    """
    if source_version >= FORMAT_VERSION:
        return fit_result

    if not isinstance(fit_result, FitResult):
        return fit_result

    if not hasattr(fit_result, "target_encoder") or fit_result.target_encoder is None:
        fit_result = dataclasses.replace(
            fit_result, target_encoder=TargetEncoder.no_op()
        )
    return fit_result


def load(path: str | Path) -> tuple[Any, Any, dict[str, Any], Any]:
    """Load Model artifacts from *path*.

    Args:
        path: Directory previously created by
            :func:`~lizyml.persistence.exporter.export`.

    Returns:
        ``(fit_result, refit_result, metadata, analysis_context)`` tuple.
        ``analysis_context`` is ``None`` for legacy artifacts that lack it.

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

    # Check format_version (H-0070: accept any version in the supported set)
    fv = metadata.get("format_version")
    if fv not in _SUPPORTED_FORMAT_VERSIONS:
        raise LizyMLError(
            code=ErrorCode.DESERIALIZATION_FAILED,
            user_message=(
                f"Unsupported format_version={fv!r}. "
                f"This version of LizyML supports "
                f"format_version in {sorted(_SUPPORTED_FORMAT_VERSIONS)}."
            ),
            context={
                "format_version": fv,
                "supported": sorted(_SUPPORTED_FORMAT_VERSIONS),
            },
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

    # H-0070: upgrade older artifacts in-memory so callers always see the
    # current FitResult contract (e.g. target_encoder field present).
    fit_result = _migrate_fit_result(fit_result, int(fv))

    # Optional: analysis_context for diagnostic APIs after load
    analysis_context = None
    ctx_pkl = src / "analysis_context.pkl"
    if ctx_pkl.exists():
        try:
            analysis_context = joblib.load(ctx_pkl)
        except Exception as exc:
            raise LizyMLError(
                code=ErrorCode.DESERIALIZATION_FAILED,
                user_message=f"Failed to load analysis_context: {exc}",
                context={"path": str(path)},
                cause=exc,
            ) from exc

    return fit_result, refit_result, metadata, analysis_context
