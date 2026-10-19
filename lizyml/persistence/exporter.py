"""Exporter — save Model artifacts to a directory.

Directory layout (format_version=3)::

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
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import joblib
import pandas as pd

from lizyml.core.category_values import encode, refusal, shown, type_name
from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.core.types.task import TaskType

if TYPE_CHECKING:
    from lizyml.core.types.fit_result import FitResult
    from lizyml.core.types.tuning_result import TuningResult
    from lizyml.estimators.base import BaseEstimatorAdapter
    from lizyml.training.results import RefitResultView

#: H-0133: 3. A pipeline state may hand LightGBM position stand-ins
#: (``stand_in_columns``); an older LizyML would read such a state as raw
#: categories and predict wrongly without an error, so it must refuse it.
FORMAT_VERSION = 3


def _state_category_values(state: Any) -> list[tuple[str, Any]]:
    """Every (column, value) a pipeline state's encoder records: each
    category and the mode (a column without categories has mode ``None``)."""
    if not isinstance(state, Mapping):
        return []
    encoder = state.get("encoder")
    if not isinstance(encoder, Mapping):
        return []
    found: list[tuple[str, Any]] = []
    categories = encoder.get("categories")
    if isinstance(categories, Mapping):
        found.extend((str(c), v) for c, cats in categories.items() for v in cats)
    modes = encoder.get("modes")
    if isinstance(modes, Mapping):
        found.extend((str(c), v) for c, v in modes.items() if v is not None)
    return found


def unsaveable_category_values(state: Any) -> list[tuple[str, Any, str]]:
    """The first value outside V of each column of a pipeline state, as
    ``(column, value, reason)``, in column order (H-0133 P5)."""
    found: dict[str, tuple[str, Any, str]] = {}
    for column, value in _state_category_values(state):
        if column in found:
            continue
        reason = refusal(value)
        if reason is not None:
            found[column] = (column, value, reason)
    return list(found.values())


def warn_unsaveable_category_values(state: Any) -> None:
    """Warn once per column whose category values ``Model.export`` and
    ``export_code`` will refuse (H-0127, H-0133 P5)."""
    for column, value, reason in unsaveable_category_values(state):
        warnings.warn(
            f"Column '{column}' holds the category value {shown(value)} of type "
            f"{type_name(value)}, which a saved model cannot hold ({reason}): "
            "fit and predict work, but Model.export and export_code will "
            "refuse this model. Convert the column's values to one of the "
            "saveable types (str, int, float, bool, bytes, Decimal, dates, "
            "datetimes, timedeltas, complex, or tuples / frozensets of them) "
            "before fit to be able to save it.",
            UserWarning,
            stacklevel=3,
        )


def _refuse_value(column: str, value: Any, reason: str, where: str) -> LizyMLError:
    """*where* is the state name (``"refit"``, ``"fold 1"``, H-0127's context
    values) or ``"declared categories"``."""
    shown_where = where if where == "declared categories" else f"{where} pipeline state"
    return LizyMLError(
        code=ErrorCode.SERIALIZATION_FAILED,
        user_message=(
            f"Model.export cannot save this model: column '{column}' "
            f"({shown_where}) "
            f"holds the category value {shown(value)} of type {type_name(value)}, "
            f"which a saved model cannot hold ({reason}). Saveable values are "
            "str, int, float, bool, bytes, Decimal, dates, datetimes, "
            "Timestamps, timedeltas, complex, the numpy scalars of these, and "
            "tuples / frozensets of them (H-0133)."
        ),
        context={
            "column": column,
            "value": shown(value),
            "type": type_name(value),
            "state": where,
        },
    )


def _saved_pipeline_states(
    fit_result: FitResult, refit_result: RefitResultView
) -> list[tuple[str, Any, BaseEstimatorAdapter]]:
    """Every pipeline state whose booster is pickled, named for the error, with
    the adapter saved with it (H-0129).

    Raises:
        LizyMLError: ``SERIALIZATION_FAILED`` when a state has no adapter to ask:
            the fold states and ``fit_result.models`` differ in number, or there
            are no per-fold states and no model.
    """
    per_fold = fit_result.pipeline_state_per_fold
    models = list(fit_result.models)
    if (per_fold is None and not models) or (
        per_fold is not None and len(per_fold) != len(models)
    ):
        raise LizyMLError(
            code=ErrorCode.SERIALIZATION_FAILED,
            user_message=(
                "Model.export cannot check this model's category values: "
                f"{'no' if per_fold is None else len(per_fold)} per-fold pipeline "
                f"state(s) and {len(models)} fold model(s) do not pair up."
            ),
            context={
                "fold_states": None if per_fold is None else len(per_fold),
                "models": len(models),
            },
        )
    named: list[tuple[str, Any, BaseEstimatorAdapter]] = [
        ("refit", refit_result.pipeline_state, refit_result.model)
    ]
    if per_fold is None:
        named.append(("last fold", fit_result.pipeline_state, models[-1]))
    else:
        named.extend(
            (f"fold {k}", state, model)
            for k, (state, model) in enumerate(zip(per_fold, models, strict=True), 1)
        )
    return named


def check_category_values(
    fit_result: FitResult,
    refit_result: RefitResultView,
    declared_categories: Mapping[str, list[Any]] | None = None,
) -> None:
    """Refuse, before anything is written, a model whose category values
    cannot be saved (H-0127, H-0133).

    Two checks, in order:

    1. Every category and mode of every saved pipeline state, and every
       declared category, must be in the closed value set V
       (:mod:`lizyml.core.category_values`), the same set ``export_code``
       accepts (H-0133 P5).
    2. Each saved pipeline state is checked by the adapter saved with it
       (``BaseEstimatorAdapter.unsaveable_categories``, H-0129), so this layer
       imports no concrete estimator. LightGBM writes each categorical
       column's categories as JSON when the booster is pickled; a column it
       cannot write is handed to it as positions (H-0133), so this check
       guards only a state built without that rule.

    Raises:
        LizyMLError: With ``SERIALIZATION_FAILED`` naming the column, the value,
            its type and the pipeline state that holds it, or a state that has
            no adapter to ask.
    """
    saved = _saved_pipeline_states(fit_result, refit_result)
    for state_name, state, _ in saved:
        for column, value in _state_category_values(state):
            reason = refusal(value)
            if reason is not None:
                raise _refuse_value(column, value, reason, state_name)
    for column, cats in (declared_categories or {}).items():
        for value in cats:
            reason = refusal(value)
            if reason is not None:
                raise _refuse_value(str(column), value, reason, "declared categories")
    for state_name, state, adapter in saved:
        found = adapter.unsaveable_categories(state)
        if not found:
            continue
        item = found[0]
        raise LizyMLError(
            code=ErrorCode.SERIALIZATION_FAILED,
            user_message=(
                f"Model.export cannot save this model: column '{item.column}' "
                f"({state_name} pipeline state) holds the category value "
                f"{item.shown} of type "
                f"{item.type_name}, which LightGBM cannot write into a saved "
                "model (it stores category values as JSON). Convert the "
                "column's values to str (or int, float, bool) and fit again."
            ),
            context={
                "column": item.column,
                "value": item.shown,
                "type": item.type_name,
                "state": state_name,
            },
        )


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
    refit_result: RefitResultView,
    config: dict[str, Any],
    task: TaskType,
    *,
    analysis_context: AnalysisContext | None = None,
    tuning: TuningResult | None = None,
    tuning_fixed_params: dict[str, Any] | None = None,
    applied_training_params: dict[str, Any] | None = None,
    applied_sample_weight: str | None = None,
    declared_categories: dict[str, list[Any]] | None = None,
    categories_checked: bool = False,
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
            means unknown and omits the key. The values are written with the
            tagged codec of :mod:`lizyml.core.category_values` (H-0133 P6);
            a value outside its set is refused before anything is written.
        categories_checked: Deprecated (H-0133) and ignored: the public
            function always runs :func:`check_category_values`, so a caller
            cannot write an artifact it would refuse. Passing ``True`` warns.
            Will be removed in v1.0.

    Raises:
        LizyMLError with SERIALIZATION_FAILED on any I/O or serialization error,
        and before anything is written when a category value cannot be saved
        (:func:`check_category_values`, H-0127).
    """
    if categories_checked:
        warnings.warn(
            "export(categories_checked=True) is deprecated and ignored: the "
            "category values are always checked before anything is written "
            "(H-0133). Will be removed in v1.0.",
            DeprecationWarning,
            stacklevel=2,
        )
    check_category_values(fit_result, refit_result, declared_categories)
    write_checked(
        path,
        fit_result,
        refit_result,
        config,
        task,
        analysis_context=analysis_context,
        tuning=tuning,
        tuning_fixed_params=tuning_fixed_params,
        applied_training_params=applied_training_params,
        applied_sample_weight=applied_sample_weight,
        declared_categories=declared_categories,
    )


def write_checked(
    path: str | Path,
    fit_result: FitResult,
    refit_result: RefitResultView,
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
    """Write the artifacts of results :func:`check_category_values` has
    already passed (``Model.export`` checks before it resolves the path).

    Internal to LizyML: not re-exported by :mod:`lizyml.persistence`, whose
    :func:`export` always checks first. Arguments as :func:`export`.

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
        if declared_categories is not None:
            # H-0133 P6: every value is in V (checked above), written with the
            # tagged codec; loader.load decodes it.
            metadata["declared_categories"] = {
                col: [encode(v) for v in cats]
                for col, cats in declared_categories.items()
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
