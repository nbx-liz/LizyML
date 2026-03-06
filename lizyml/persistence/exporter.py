"""Exporter — save Model artifacts to a directory.

Directory layout (format_version=1)::

    {path}/
        metadata.json        — human-readable metadata + version info
        fit_result.pkl       — FitResult (joblib compressed)
        refit_model.pkl      — RefitResult (joblib compressed)
        analysis_context.pkl — (optional) y_true + X for diagnostic APIs

Security note: pickle/joblib files must only be loaded from trusted sources.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import joblib
import pandas as pd

from lizyml.core.exceptions import ErrorCode, LizyMLError

if TYPE_CHECKING:
    from lizyml.core.types.fit_result import FitResult
    from lizyml.training.refit_trainer import RefitResult

FORMAT_VERSION = 1


@dataclass
class AnalysisContext:
    """Data needed for diagnostic APIs after Model.load()."""

    y_true: pd.Series
    X_for_explain: pd.DataFrame


def export(
    path: str | Path,
    fit_result: FitResult,
    refit_result: RefitResult,
    config: dict[str, Any],
    task: str,
    *,
    analysis_context: AnalysisContext | None = None,
) -> None:
    """Serialize Model artifacts to *path*.

    Args:
        path: Output directory path (created if it does not exist).
        fit_result: Completed CV training output.
        refit_result: Full-data refit output used for inference.
        config: Normalized config dict (from ``LizyMLConfig.model_dump()``).
        task: ML task string (``"regression"``, ``"binary"``, ``"multiclass"``).
        analysis_context: Optional y_true and X data for diagnostic APIs
            after ``Model.load()``.

    Raises:
        LizyMLError with SERIALIZATION_FAILED on any I/O or serialization error.
    """
    out = Path(path)
    try:
        out.mkdir(parents=True, exist_ok=True)

        metadata: dict[str, Any] = {
            "format_version": FORMAT_VERSION,
            "lizyml_version": fit_result.run_meta.lizyml_version,
            "python_version": fit_result.run_meta.python_version,
            "timestamp": fit_result.run_meta.timestamp,
            "run_id": fit_result.run_meta.run_id,
            "config": config,
            "metrics": fit_result.metrics,
            "feature_names": fit_result.feature_names,
            "task": task,
        }
        (out / "metadata.json").write_text(
            json.dumps(metadata, indent=2, default=str), encoding="utf-8"
        )

        joblib.dump(fit_result, out / "fit_result.pkl", compress=3)
        joblib.dump(refit_result, out / "refit_model.pkl", compress=3)

        if analysis_context is not None:
            joblib.dump(analysis_context, out / "analysis_context.pkl", compress=3)

    except LizyMLError:
        raise
    except Exception as exc:
        raise LizyMLError(
            code=ErrorCode.SERIALIZATION_FAILED,
            user_message=f"Failed to export model to '{path}': {exc}",
            context={"path": str(path)},
            cause=exc,
        ) from exc
