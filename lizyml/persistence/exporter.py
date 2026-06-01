"""Exporter — save Model artifacts to a directory.

Directory layout (format_version=2)::

    {path}/
        metadata.json        — human-readable metadata + version info +
                               SHA-256 checksums of each .pkl (H-0083)
        fit_result.pkl       — FitResult (joblib compressed)
        refit_model.pkl      — RefitResult (joblib compressed)
        analysis_context.pkl — (optional) y_true + X for diagnostic APIs

Security note: pickle/joblib files must only be loaded from trusted sources.
The SHA-256 ``checksums`` in metadata.json bind the validated metadata to the
.pkl bytes so that tampering/corruption is detected on load (it does not make
pickle safe against a fully trusted-but-malicious producer). The field is
additive: artifacts without it (pre-H-0083) still load (FORMAT_VERSION=2).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import joblib
import pandas as pd

from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.core.types.task import TaskType

if TYPE_CHECKING:
    from lizyml.core.types.fit_result import FitResult
    from lizyml.training.refit_trainer import RefitResult

FORMAT_VERSION = 2

#: Checksum algorithm recorded in ``metadata.json`` and verified on load.
CHECKSUM_ALGORITHM = "sha256"


def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 digest of *path*'s bytes (H-0083)."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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
    task: TaskType,
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

        # Serialize payloads first so their bytes can be hashed into metadata
        # (integrity binding verified on load — H-0083).
        joblib.dump(fit_result, out / "fit_result.pkl", compress=3)
        joblib.dump(refit_result, out / "refit_model.pkl", compress=3)
        pkl_names = ["fit_result.pkl", "refit_model.pkl"]

        if analysis_context is not None:
            joblib.dump(analysis_context, out / "analysis_context.pkl", compress=3)
            pkl_names.append("analysis_context.pkl")

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
            "checksums": {
                "algorithm": CHECKSUM_ALGORITHM,
                "files": {name: sha256_file(out / name) for name in pkl_names},
            },
        }
        (out / "metadata.json").write_text(
            json.dumps(metadata, indent=2, default=str), encoding="utf-8"
        )

    except LizyMLError:
        raise
    except Exception as exc:
        raise LizyMLError(
            code=ErrorCode.SERIALIZATION_FAILED,
            user_message=f"Failed to export model to '{path}': {exc}",
            context={"path": str(path)},
            cause=exc,
        ) from exc
