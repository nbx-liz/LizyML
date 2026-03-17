"""Metric filtering and calibrated metrics assembly (H-0054).

Extracted from model.py to keep the facade under 800 lines.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pandas as pd

from lizyml.core.types.fit_result import FitResult
from lizyml.evaluation.evaluator import Evaluator


def _has_metric_content(filtered: dict[str, Any]) -> bool:
    """Check if a filtered metrics branch has any non-empty data."""
    for v in filtered.values():
        if isinstance(v, dict) and v:
            return True
        if isinstance(v, list) and any(isinstance(d, dict) and d for d in v):
            return True
    return False


def filter_metrics(metrics_dict: dict[str, Any], keep: set[str]) -> dict[str, Any]:
    """Return a copy of *metrics_dict* with only *keep* metric names retained.

    Works recursively on the nested
    ``{"raw": {"oof": {...}, ...}, "calibrated": {...}}``
    structure produced by :class:`~lizyml.evaluation.evaluator.Evaluator`.
    """
    result: dict[str, Any] = {}
    for top_key, top_val in metrics_dict.items():
        if not isinstance(top_val, dict):
            result[top_key] = top_val
            continue
        filtered_top: dict[str, Any] = {}
        for sub_key, sub_val in top_val.items():
            if sub_key in ("if_per_fold", "oof_per_fold"):
                # List of per-fold dicts
                filtered_top[sub_key] = [
                    {m: v for m, v in fold.items() if m in keep} for fold in sub_val
                ]
            elif isinstance(sub_val, dict):
                filtered_top[sub_key] = {m: v for m, v in sub_val.items() if m in keep}
            else:
                filtered_top[sub_key] = sub_val
        # Drop branches where all sub-dicts are empty after filtering
        if _has_metric_content(filtered_top):
            result[top_key] = filtered_top
    return result


def assemble_calibrated_metrics(
    fit_result: FitResult,
    y: pd.Series,
    metric_names: list[str],
    evaluator: Evaluator,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    """Assemble calibrated metrics branch if a calibrator is present.

    This is a Facade concern (H-0052): the Evaluator only produces raw metrics,
    and calibrated OOF assembly is done here.

    Returns:
        Updated metrics dict with ``"calibrated"`` key added when applicable.
    """
    if fit_result.calibrator is None:
        return metrics

    from lizyml.calibration.cross_fit import CalibrationResult

    if not isinstance(fit_result.calibrator, CalibrationResult):
        return metrics

    cal_oof = fit_result.calibrator.calibrated_oof
    # H-0058: calibration cross-fit reuses outer splits, so coverage
    # is identical to raw OOF.  No splits replacement needed.
    cal_fr = dataclasses.replace(
        fit_result,
        oof_pred=cal_oof,
    )
    cal_result = evaluator.evaluate(cal_fr, y, metric_names)
    return {**metrics, "calibrated": {"oof": cal_result["raw"]["oof"]}}
