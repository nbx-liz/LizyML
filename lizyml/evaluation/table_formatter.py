"""Evaluation table formatter.

Converts the nested metrics dict produced by
:class:`~lizyml.evaluation.evaluator.Evaluator` into a ``pd.DataFrame``.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def format_metrics_table(metrics: dict[str, Any]) -> pd.DataFrame:
    """Convert a structured metrics dict to a DataFrame.

    Args:
        metrics: Nested dict with ``"raw"`` (and optionally ``"calibrated"``)
            keys, as produced by :meth:`Evaluator.evaluate`.

    Returns:
        DataFrame where rows are metric names and columns are
        ``if_mean``, ``oof``, ``fold_0`` … ``fold_N-1``, and optionally
        ``cal_oof`` when calibrated metrics are present.
    """
    raw = metrics.get("raw", {})
    oof: dict[str, float] = raw.get("oof", {})
    if_mean: dict[str, float] = raw.get("if_mean", {})
    if_per_fold: list[dict[str, float]] = raw.get("if_per_fold", [])

    metric_names = list(oof.keys()) or list(if_mean.keys())
    if not metric_names:
        return pd.DataFrame()

    data: dict[str, list[float]] = {
        "if_mean": [if_mean.get(m, float("nan")) for m in metric_names],
        "oof": [oof.get(m, float("nan")) for m in metric_names],
    }

    for fold_idx, fold_dict in enumerate(if_per_fold):
        col = f"fold_{fold_idx}"
        data[col] = [fold_dict.get(m, float("nan")) for m in metric_names]

    # Calibrated OOF (binary with calibrator)
    calibrated = metrics.get("calibrated", {})
    cal_oof: dict[str, float] = calibrated.get("oof", {})
    if cal_oof:
        data["cal_oof"] = [cal_oof.get(m, float("nan")) for m in metric_names]

    df = pd.DataFrame(data, index=metric_names)
    df.index.name = "metric"
    return df
