"""Evaluator — computes structured metrics from a FitResult."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

from lizyml.core.types.fit_result import FitResult
from lizyml.metrics.base import BaseMetric
from lizyml.metrics.registry import get_metrics_for_task

TaskType = Literal["regression", "binary", "multiclass"]


def _pred_for_metric(
    metric: BaseMetric,
    raw_pred: np.ndarray,
    task: TaskType,
) -> np.ndarray:
    """Return the appropriate prediction array for *metric*.

    - ``needs_proba=True``: return ``raw_pred`` as-is.
    - ``needs_proba=False``: binarise (binary) or argmax (multiclass) as needed.
    """
    if metric.needs_proba:
        return raw_pred
    if task == "binary":
        return (raw_pred >= 0.5).astype(int)
    if task == "multiclass":
        result: np.ndarray = raw_pred.argmax(axis=1)
        return result
    return raw_pred  # regression


def _compute_metrics(
    metrics: list[BaseMetric],
    y_true: np.ndarray,
    y_pred: np.ndarray,
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
                "oof":         {metric_name: float, ...},
                "if_mean":     {metric_name: float, ...},
                "if_per_fold": [{metric_name: float}, ...],
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
        y: pd.Series | np.ndarray,
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
            if_mean[m.name] = float(np.mean([fold[m.name] for fold in if_per_fold]))

        result: dict[str, Any] = {
            "raw": {
                "oof": oof_scores,
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
