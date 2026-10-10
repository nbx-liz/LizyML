"""Exporter — save Model artifacts to a directory.

Directory layout (format_version=2)::

    {path}/
        metadata.json        — human-readable metadata + version info +
                               SHA-256 checksums of each .pkl (H-0083) +
                               the training overlay the fit applied (H-0109)
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
    from lizyml.core.types.tuning_result import TuningResult
    from lizyml.training.refit_trainer import RefitResult

FORMAT_VERSION = 2

_JSON_SCALARS = (str, int, float, bool)


def _plain_declared(
    declared: dict[str, list[Any]] | None,
) -> dict[str, list[Any]] | None:
    """The declared-categories record as typed JSON values, or ``None``.

    numpy scalars become the Python value their ``.item()`` returns when it is
    equal; any other value makes the whole record ``None`` (omitted), never a
    ``str`` (H-0120 amendment 4).
    """
    if declared is None:
        return None
    out: dict[str, list[Any]] = {}
    for col, cats in declared.items():
        values: list[Any] = []
        for value in cats:
            plain = value.item() if hasattr(value, "item") else value
            if type(plain) not in _JSON_SCALARS or plain != value:
                return None
            values.append(plain)
        out[col] = values
    return out


def _tuning_metadata(tuning: TuningResult) -> dict[str, Any]:
    """Serialize the tuned-param overlay for ``metadata.json`` (H-0086, #215).

    Only the values ``Model._merge_params`` needs to reproduce the tuned fit are
    recorded (``best_*`` params + score/metric/direction). The full trial list
    and optuna study are intentionally omitted — restoring those (for a complete
    ``tune(resume=True)`` from a loaded model) is a separate follow-up.
    """
    return {
        "best_model_params": dict(tuning.best_model_params),
        "best_smart_params": dict(tuning.best_smart_params),
        "best_training_params": dict(tuning.best_training_params),
        "best_score": tuning.best_score,
        "metric_name": tuning.metric_name,
        "direction": tuning.direction,
    }


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
    tuning: TuningResult | None = None,
    tuning_fixed_params: dict[str, Any] | None = None,
    applied_training_params: dict[str, Any] | None = None,
    applied_sample_weight: str | None = None,
    declared_categories: dict[str, list[Any]] | None = None,
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
        tuning: Optional tuning result. When present, the tuned-param overlay
            is recorded under ``metadata["tuning"]`` so a re-``fit()`` after
            ``Model.load()`` reproduces the tuned params (H-0086, #215).
        tuning_fixed_params: Effective fixed policy of the successful tuning
            round. None omits metadata for legacy fallback; {} records no defaults.
        applied_training_params: The ``best_training_params`` overlay the fit
            that produced *fit_result* applied -- ``{}`` when it applied none.
            Recorded under ``metadata["applied_training_params"]`` (H-0109),
            because the ``tuning`` block is the model's *current* tuning result
            and may be one no fit consumed. ``None`` means unknown (a model
            loaded from an artifact without the record) and omits the key, so
            re-exporting such a model does not invent a record.
        applied_sample_weight: The row-weight rule that fit's refit applied,
            ``"balanced"`` or ``"none"``, recorded under
            ``metadata["applied_sample_weight"]`` (H-0120 amendment 1).
            ``None`` means unknown and omits the key, as above.
        declared_categories: The features the fit's input frame held as
            ``category``, with their categories, recorded under
            ``metadata["declared_categories"]`` (H-0120 amendment 4). ``None``
            means unknown and omits the key. A record holding a value JSON
            cannot return with its type is omitted too rather than written as
            ``str``: ``export_code`` refuses such a model anyway, because the
            encoder holds the same values.

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
        if tuning is not None:
            # Additive (H-0086, #215): absent for non-tuned models and pre-#215
            # artifacts, which load with ``_tuning_result = None`` as before.
            metadata["tuning"] = _tuning_metadata(tuning)
            if tuning_fixed_params is not None:
                metadata["tuning"]["fixed_params"] = dict(tuning_fixed_params)
        if applied_training_params is not None:
            metadata["applied_training_params"] = dict(applied_training_params)
        if applied_sample_weight is not None:
            metadata["applied_sample_weight"] = applied_sample_weight
        declared = _plain_declared(declared_categories)
        if declared is not None:
            metadata["declared_categories"] = declared
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
