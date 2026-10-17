"""Evaluator — computes structured metrics from a FitResult."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.core.types.fit_result import FitResult
from lizyml.core.types.oof_coverage import compute_oof_valid_mask
from lizyml.core.types.task import TaskType
from lizyml.metrics.base import BaseMetric
from lizyml.metrics.registry import MetricEntry, get_metrics_for_task

__all__ = ["Evaluator", "TaskType"]


def _normalize_multiclass_proba(
    pred: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Row-wise normalize multiclass predictions to sum to 1.0.

    Required when the objective is ``multiclassova`` (independent sigmoid
    per class).  For ``multiclass`` (softmax), predictions already sum to
    1.0 and this operation is idempotent.
    """
    row_sums = pred.sum(axis=1, keepdims=True)
    # Guard against all-zero rows (degenerate edge case).
    # Positive values always sum to a positive number, so exact == 0.0
    # only fires when every element in the row is 0.0.
    row_sums = np.where(row_sums == 0.0, 1.0, row_sums)
    normalized: npt.NDArray[np.float64] = pred / row_sums
    return normalized


def _pred_for_metric(
    metric: BaseMetric,
    raw_pred: npt.NDArray[np.float64],
    task: TaskType,
) -> npt.NDArray[Any]:
    """Return the appropriate prediction array for *metric*.

    - ``needs_proba=True`` and ``needs_simplex=True``:
      - multiclass 2-D predictions are row-normalised so that metrics
        receiving probabilities always see a valid distribution (sum = 1).
        This is necessary for ``multiclassova`` (independent sigmoid),
        and idempotent for ``multiclass`` (softmax).
    - ``needs_proba=True`` and ``needs_simplex=False``:
      - Per-class OvR metrics (e.g. AUCPR, Brier) receive raw predictions.
    - ``needs_proba=False``: binarise (binary) or argmax (multiclass).
    """
    if metric.needs_proba:
        if task == "multiclass" and raw_pred.ndim == 2 and metric.needs_simplex:
            return _normalize_multiclass_proba(raw_pred)
        return raw_pred
    if task == "binary":
        return (raw_pred >= 0.5).astype(int)
    if task == "multiclass":
        result: npt.NDArray[np.intp] = raw_pred.argmax(axis=1)
        return result
    return raw_pred  # regression


def _compute_metrics(
    metrics: list[BaseMetric],
    y_true: npt.NDArray[Any],
    y_pred: npt.NDArray[np.float64],
    task: TaskType,
    y_train: npt.NDArray[Any] | None = None,
) -> dict[str, float]:
    """Compute *metrics*; a metric that needs training targets is first bound
    to *y_train*, the targets of the rows the scored model was fitted on
    (H-0132). The caller never passes the scored rows as *y_train*."""
    result: dict[str, float] = {}
    for m in metrics:
        pred = _pred_for_metric(m, y_pred, task)
        metric = m.bind_train(np.asarray(y_train)) if m.needs_train_target else m
        result[m.name] = metric(y_true, pred)
    return result


def _refuse_overlapping_folds(
    metrics: list[BaseMetric],
    outer: list[tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]],
    n_rows: int,
) -> None:
    """A metric that needs training targets scales each OOF row by the fold
    that validated it, so each covered row must belong to exactly one
    validation fold (H-0132 policy 4). Refuse instead of picking a fold."""
    needing = [m.name for m in metrics if m.needs_train_target]
    if not needing:
        return
    counts = np.zeros(n_rows, dtype=np.intp)
    for _, valid_idx in outer:
        np.add.at(counts, valid_idx, 1)
    overlapping = np.flatnonzero(counts > 1)
    if len(overlapping):
        raise LizyMLError(
            code=ErrorCode.UNSUPPORTED_METRIC,
            user_message=(
                f"Metric '{needing[0]}' scales each out-of-fold row by the fold "
                f"that validated it, but {len(overlapping)} row(s) are in more "
                f"than one validation fold (first: row {int(overlapping[0])})."
            ),
            context={
                "metric": needing[0],
                "overlapping_rows": int(len(overlapping)),
                "first_row": int(overlapping[0]),
            },
        )


class Evaluator:
    """Compute structured evaluation metrics from a :class:`FitResult`.

    The output structure is fixed::

        {
            "raw": {
                "oof":          {metric_name: float, ...},
                "oof_per_fold": [{metric_name: float}, ...],
                "if_mean":      {metric_name: float, ...},
                "if_per_fold":  [{metric_name: float}, ...],
                "oof_coverage": float,   # fraction of rows covered by a
                                         # validation fold (H-0057)
            },
            "calibrated": { ... }   # populated only when calibrator is set
        }

    Metric computation is centralised here; no metric calculation is done
    outside this class.

    Args:
        task: ML task type; used for task-compatibility validation.
    """

    def __init__(self, task: TaskType = "regression") -> None:
        self.task = task

    def evaluate(
        self,
        fit_result: FitResult,
        y: pd.Series | npt.NDArray[Any],
        metric_entries: list[MetricEntry],
    ) -> dict[str, Any]:
        """Compute OOF, IF-per-fold, and IF-mean metrics.

        Args:
            fit_result: Completed CV training output.
            y: Ground-truth target for the full dataset (same order as
                ``fit_result.oof_pred``).
            metric_entries: Metric names or parameterised ``MetricEntry``
                dicts to compute.  Must be compatible with the task this
                :class:`Evaluator` was constructed for.

        Returns:
            Nested dict with ``"raw"`` (and ``"calibrated"`` when applicable).
        """
        metrics = get_metrics_for_task(metric_entries, self.task)
        y_arr = np.asarray(y)

        # --- OOF coverage mask (H-0057) --------------------------------------
        oof_mask = compute_oof_valid_mask(fit_result.splits.outer, len(y_arr))
        covered_pred = fit_result.oof_pred[oof_mask]

        # Assert no NaN in covered rows — NaN here indicates a pipeline bug,
        # not a structurally uncovered row (e.g. TimeSeriesCV first period).
        _nan_flags = np.isnan(covered_pred)
        # For 2-D predictions (multiclass), collapse to per-row bool.
        nan_in_covered = (
            _nan_flags if _nan_flags.ndim == 1 else np.asarray(_nan_flags.any(axis=1))
        )
        if nan_in_covered.any():
            nan_count = int(nan_in_covered.sum())
            nan_indices = np.where(oof_mask)[0][nan_in_covered]
            raise LizyMLError(
                code=ErrorCode.EVALUATION_FAILED,
                user_message=(
                    f"OOF predictions contain NaN in {nan_count} row(s) "
                    f"covered by validation folds (indices: "
                    f"{nan_indices[:5].tolist()}{'...' if nan_count > 5 else ''}). "
                    f"This indicates a bug in the training pipeline."
                ),
                context={
                    "nan_count": nan_count,
                    "nan_indices": nan_indices[:20].tolist(),
                    "task": self.task,
                },
            )

        oof_coverage = float(oof_mask.sum()) / len(y_arr) if len(y_arr) > 0 else 0.0

        _refuse_overlapping_folds(metrics, fit_result.splits.outer, len(y_arr))

        # --- Per-fold OOF metrics (valid_idx) --------------------------------
        # A metric that needs training targets is bound to the fold's
        # outer-train rows (H-0132), never to the validated rows.
        oof_per_fold: list[dict[str, float]] = []
        for train_idx, valid_idx in fit_result.splits.outer:
            y_valid = y_arr[valid_idx]
            oof_valid_pred = fit_result.oof_pred[valid_idx]
            oof_per_fold.append(
                _compute_metrics(
                    metrics,
                    y_valid,
                    oof_valid_pred,
                    self.task,
                    y_train=y_arr[train_idx],
                )
            )

        # --- OOF metrics (covered rows only) ----------------------------------
        # Metrics that need training targets: each covered row is scaled by the
        # fold that validated it, i.e. the fold-size-weighted mean of the
        # per-fold values over the covered rows (H-0132 policy 4).
        pooled = [m for m in metrics if not m.needs_train_target]
        pooled_scores = _compute_metrics(
            pooled, y_arr[oof_mask], covered_pred, self.task
        )
        sizes = np.array([len(v) for _, v in fit_result.splits.outer], dtype=float)
        oof_scores: dict[str, float] = {}
        for m in metrics:
            if m.needs_train_target:
                values = np.array([fold[m.name] for fold in oof_per_fold])
                filled = sizes > 0
                oof_scores[m.name] = float(
                    np.sum(values[filled] * sizes[filled]) / np.sum(sizes[filled])
                )
            else:
                oof_scores[m.name] = pooled_scores[m.name]

        # --- Per-fold IF metrics ---------------------------------------------
        if_per_fold: list[dict[str, float]] = []
        for k, (train_idx, _) in enumerate(fit_result.splits.outer):
            y_train = y_arr[train_idx]
            fold_pred = fit_result.if_pred_per_fold[k]
            fold_scores = _compute_metrics(
                metrics, y_train, fold_pred, self.task, y_train=y_train
            )
            if_per_fold.append(fold_scores)

        # --- IF mean ---------------------------------------------------------
        if_mean: dict[str, float] = {
            m.name: float(np.mean([fold[m.name] for fold in if_per_fold]))
            for m in metrics
        }

        return {
            "raw": {
                "oof": oof_scores,
                "oof_per_fold": oof_per_fold,
                "if_mean": if_mean,
                "if_per_fold": if_per_fold,
                "oof_coverage": oof_coverage,
            }
        }
