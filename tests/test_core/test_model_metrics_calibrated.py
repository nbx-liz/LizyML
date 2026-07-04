"""Regression tests for calibrated metric assembly (#7 + code-review fixes).

- ``metrics['calibrated']`` must include ``oof_per_fold`` and must NOT include
  IF metrics (IF calibration would leak).
- ``filter_metrics`` must drop branches left empty after filtering.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from lizyml.calibration.cross_fit import CalibrationResult
from lizyml.core._model_metrics import assemble_calibrated_metrics, filter_metrics
from lizyml.core.types.artifacts import DataFingerprint, SplitIndices
from lizyml.core.types.fit_result import FitResult
from lizyml.evaluation.evaluator import Evaluator


def _make_fit_result_with_calibrator() -> tuple[FitResult, pd.Series]:
    n = 20
    rng = np.random.default_rng(42)
    oof = rng.random(n)
    cal_oof = np.clip(oof + rng.normal(0, 0.01, n), 0, 1)
    outer_splits = [
        (np.arange(10, dtype=np.intp), np.arange(10, 20, dtype=np.intp)),
        (np.arange(10, 20, dtype=np.intp), np.arange(10, dtype=np.intp)),
    ]
    splits = SplitIndices(
        outer=outer_splits,
        inner=[(np.array([], dtype=np.intp), np.array([], dtype=np.intp))] * 2,
        calibration=None,
    )
    cal_result = CalibrationResult(
        c_final=MagicMock(),
        calibrated_oof=cal_oof,
        method="platt",
        split_indices=outer_splits,
    )
    fr = FitResult(
        oof_pred=oof,
        if_pred_per_fold=[rng.random(10), rng.random(10)],
        metrics={},
        models=[MagicMock(), MagicMock()],
        history=[{}, {}],
        feature_names=["f1"],
        dtypes={"f1": "float64"},
        categorical_features=[],
        splits=splits,
        data_fingerprint=DataFingerprint(row_count=n, column_hash="abc"),
        pipeline_state={},
        calibrator=cal_result,
        run_meta=MagicMock(),
        oof_raw_scores=None,
    )
    return fr, pd.Series((oof > 0.5).astype(int))


class TestCalibratedMetricsOofPerFold:
    def test_calibrated_has_oof_per_fold(self) -> None:
        fr, y = _make_fit_result_with_calibrator()
        evaluator = Evaluator(task="binary")
        raw_metrics = evaluator.evaluate(fr, y, ["auc", "logloss"])
        result = assemble_calibrated_metrics(
            fr, y, ["auc", "logloss"], evaluator, raw_metrics
        )
        assert "calibrated" in result
        assert "oof" in result["calibrated"]
        assert "oof_per_fold" in result["calibrated"]
        assert isinstance(result["calibrated"]["oof_per_fold"], list)
        assert len(result["calibrated"]["oof_per_fold"]) == 2

    def test_calibrated_excludes_if_metrics(self) -> None:
        fr, y = _make_fit_result_with_calibrator()
        evaluator = Evaluator(task="binary")
        raw_metrics = evaluator.evaluate(fr, y, ["auc"])
        result = assemble_calibrated_metrics(fr, y, ["auc"], evaluator, raw_metrics)
        assert "if_mean" not in result.get("calibrated", {})
        assert "if_per_fold" not in result.get("calibrated", {})


class TestFilterMetricsNoBranches:
    def test_no_empty_calibrated_branch(self) -> None:
        metrics = {
            "raw": {"oof": {"rmse": 0.5, "mae": 0.3}, "if_mean": {"rmse": 0.4}},
            "calibrated": {"oof": {"logloss": 0.2}},
        }
        result = filter_metrics(metrics, {"rmse"})
        if "calibrated" in result:
            for k, v in result["calibrated"].items():
                if isinstance(v, dict):
                    assert len(v) > 0, f"Empty branch: calibrated.{k}"
