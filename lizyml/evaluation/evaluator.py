"""Evaluator — computes structured metrics from a FitResult."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from lizyml.core.types.fit_result import FitResult
from lizyml.metrics.base import BaseMetric
from lizyml.metrics.registry import get_metrics_for_task

TaskType = Literal["regression", "binary", "multiclass"]


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
) -> dict[str, float]:
    result: dict[str, float] = {}
    for m in metrics:
        pred = _pred_for_metric(m, y_pred, task)
        result[m.name] = m(y_true, pred)
    return result


class Evaluator:
    """Compute structured evaluation metrics from a :class:`FitResult`.

    The output structure is fixed::

        {
            "raw": {
                "oof":          {metric_name: float, ...},
                "oof_per_fold": [{metric_name: float}, ...],
                "if_mean":      {metric_name: float, ...},
                "if_per_fold":  [{metric_name: float}, ...],
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
        metric_names: list[str],
    ) -> dict[str, Any]:
        """Compute OOF, IF-per-fold, and IF-mean metrics.

        Args:
            fit_result: Completed CV training output.
            y: Ground-truth target for the full dataset (same order as
                ``fit_result.oof_pred``).
            metric_names: Names of metrics to compute.  Must be compatible
                with the task this :class:`Evaluator` was constructed for.

        Returns:
            Nested dict with ``"raw"`` (and ``"calibrated"`` when applicable).
        """
        metrics = get_metrics_for_task(metric_names, self.task)
        y_arr = np.asarray(y)

        # --- OOF metrics -----------------------------------------------------
        oof_scores = _compute_metrics(metrics, y_arr, fit_result.oof_pred, self.task)

        # --- Per-fold OOF metrics (valid_idx) --------------------------------
        oof_per_fold: list[dict[str, float]] = []
        for _, valid_idx in fit_result.splits.outer:
            y_valid = y_arr[valid_idx]
            oof_valid_pred = fit_result.oof_pred[valid_idx]
            oof_per_fold.append(
                _compute_metrics(metrics, y_valid, oof_valid_pred, self.task)
            )

        # --- Per-fold IF metrics ---------------------------------------------
        if_per_fold: list[dict[str, float]] = []
        for k, (train_idx, _) in enumerate(fit_result.splits.outer):
            y_train = y_arr[train_idx]
            fold_pred = fit_result.if_pred_per_fold[k]
            fold_scores = _compute_metrics(metrics, y_train, fold_pred, self.task)
            if_per_fold.append(fold_scores)

        # --- IF mean ---------------------------------------------------------
        if_mean: dict[str, float] = {}
        for m in metrics:
            vals = [fold[m.name] for fold in if_per_fold]
            if_mean[m.name] = float(np.mean(vals))

        result: dict[str, Any] = {
            "raw": {
                "oof": oof_scores,
                "oof_per_fold": oof_per_fold,
                "if_mean": if_mean,
                "if_per_fold": if_per_fold,
            }
        }

        # --- Calibrated metrics -------------------------------------------
        if fit_result.calibrator is not None:
            from lizyml.calibration.cross_fit import CalibrationResult

            if isinstance(fit_result.calibrator, CalibrationResult):
                cal_oof = fit_result.calibrator.calibrated_oof
                cal_oof_scores = _compute_metrics(metrics, y_arr, cal_oof, self.task)
                result["calibrated"] = {
                    "oof": cal_oof_scores,
                }

        return result
