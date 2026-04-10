"""Regression tests for issues #5, #6, #7 discovered 2026-04-10.

#5: RefitTrainer pipeline fitted on all data (inner-valid included).
#6: cross_fit_calibrate passes NaN val_idx to calibrator.predict.
#7: calibrated metrics missing oof_per_fold key.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from lizyml.calibration.cross_fit import (
    CalibrationResult,
    cross_fit_calibrate,
)
from lizyml.core._model_metrics import assemble_calibrated_metrics
from lizyml.core.types.fit_result import FitResult
from lizyml.evaluation.evaluator import Evaluator
from lizyml.features.pipeline_base import BaseFeaturePipeline
from lizyml.training.inner_valid import HoldoutInnerValid
from lizyml.training.refit_trainer import RefitTrainer

# ---------------------------------------------------------------------------
# #5: RefitTrainer — pipeline must not fit on inner-valid rows
# ---------------------------------------------------------------------------


class SpyPipeline(BaseFeaturePipeline):
    """Pipeline that records which rows it was fitted on."""

    def __init__(self) -> None:
        self.fit_n_rows: int | None = None
        self._cols: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> SpyPipeline:
        self.fit_n_rows = len(X)
        self._cols = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X

    def get_state(self) -> dict[str, Any]:
        return {"cols": self._cols}

    def load_state(self, state: dict[str, Any]) -> SpyPipeline:
        self._cols = state["cols"]
        return self


class StubEstimator:
    """Minimal estimator stub for RefitTrainer tests."""

    def __init__(self) -> None:
        self.eval_results: dict[str, list[float]] = {}
        self.best_iteration: int | None = None

    def set_categorical_features(self, _: Any) -> None:
        pass

    def update_params(self, _: dict[str, Any]) -> None:
        pass

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_valid: pd.DataFrame | None = None,
        y_valid: pd.Series | None = None,
    ) -> None:
        pass

    def predict(self, X: pd.DataFrame) -> npt.NDArray[np.float64]:
        return np.zeros(len(X), dtype=np.float64)


class TestRefitTrainerPipelineLeakage:
    """Pipeline must be fitted on inner-train only, not all data."""

    def test_pipeline_fit_excludes_inner_valid_rows(self) -> None:
        """When inner-valid is used, the initial pipeline.fit must
        be called on inner-train rows only (not the full dataset).
        The final pipeline_state should be from a full-data refit.
        """
        n = 100
        X = pd.DataFrame({"f1": np.arange(n, dtype=float)})
        y = pd.Series(np.random.default_rng(0).random(n))

        spy_pipelines: list[SpyPipeline] = []

        def pipeline_factory() -> SpyPipeline:
            p = SpyPipeline()
            spy_pipelines.append(p)
            return p

        inner_valid = HoldoutInnerValid(ratio=0.2, random_state=42)
        trainer = RefitTrainer(
            inner_valid=inner_valid,
            pipeline_factory=pipeline_factory,
            estimator_factory=StubEstimator,  # type: ignore[arg-type]
            task="regression",
        )

        result = trainer.fit(X, y)

        # Two pipelines: inner-train fit + full-data fit
        assert len(spy_pipelines) == 2, (
            f"Expected 2 pipeline instances, got {len(spy_pipelines)}"
        )
        # First pipeline: fitted on inner-train only
        assert spy_pipelines[0].fit_n_rows is not None
        assert spy_pipelines[0].fit_n_rows < n, (
            f"First pipeline fitted on {spy_pipelines[0].fit_n_rows} "
            f"rows, expected < {n} (inner-train only)"
        )
        # Second pipeline: fitted on full data
        assert spy_pipelines[1].fit_n_rows == n, (
            f"Second pipeline fitted on {spy_pipelines[1].fit_n_rows} "
            f"rows, expected {n} (full data)"
        )
        # pipeline_state comes from full-data fit
        assert result.pipeline_state == {"cols": ["f1"]}


# ---------------------------------------------------------------------------
# #6: cross_fit_calibrate — NaN in val_idx must not reach calibrator
# ---------------------------------------------------------------------------


class StubCalibrator:
    """Calibrator that records inputs and tracks NaN."""

    name: str = "stub"

    def __init__(self) -> None:
        self.predict_received_nan: bool = False

    def fit(
        self,
        scores: npt.NDArray[np.float64],
        y: npt.NDArray[Any],
    ) -> None:
        pass

    def predict(self, scores: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        if np.any(np.isnan(scores)):
            self.predict_received_nan = True
        # Return scores as-is (identity calibration)
        return scores.copy()


class TestCrossFitNaNGuard:
    """cross_fit_calibrate must not pass NaN to calibrator.predict."""

    def test_nan_val_rows_use_fallback(self) -> None:
        """When val_idx includes rows with NaN OOF scores,
        those rows must use fallback instead of cal.predict.
        """
        n = 10
        oof_scores = np.array([np.nan, np.nan, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
        y = np.array([0, 0, 0, 1, 0, 1, 0, 1, 1, 1])
        fallback = np.full(n, 0.5)

        # Split where val_idx includes NaN rows (indices 0, 1)
        split_indices = [
            (
                np.array([2, 3, 4, 5, 6], dtype=np.intp),
                np.array([0, 1, 7, 8, 9], dtype=np.intp),
            ),
            (
                np.array([0, 1, 7, 8, 9], dtype=np.intp),
                np.array([2, 3, 4, 5, 6], dtype=np.intp),
            ),
        ]

        calibrators: list[StubCalibrator] = []

        def cal_factory() -> StubCalibrator:
            c = StubCalibrator()
            calibrators.append(c)
            return c

        result = cross_fit_calibrate(
            oof_scores,
            y,
            cal_factory,  # type: ignore[arg-type]
            split_indices=split_indices,
            oof_pred=fallback,
        )

        # No calibrator should have received NaN in predict
        predict_cals = calibrators[:-1]  # last is c_final
        for i, cal in enumerate(predict_cals):
            assert not cal.predict_received_nan, (
                f"Fold {i} calibrator received NaN in predict(). "
                "NaN val rows should use fallback."
            )

        # NaN rows should have fallback value
        assert result.calibrated_oof[0] == pytest.approx(0.5)
        assert result.calibrated_oof[1] == pytest.approx(0.5)

        # Covered rows should have calibrated values (not NaN)
        for i in range(2, 10):
            assert not np.isnan(result.calibrated_oof[i]), (
                f"Row {i} should be calibrated, got NaN"
            )


# ---------------------------------------------------------------------------
# #7: calibrated metrics must include oof_per_fold
# ---------------------------------------------------------------------------


class TestCalibratedMetricsOofPerFold:
    """calibrated branch must include oof_per_fold."""

    @staticmethod
    def _make_fit_result_with_calibrator() -> tuple[FitResult, pd.Series]:
        """Create a minimal FitResult with CalibrationResult."""
        n = 20
        rng = np.random.default_rng(42)
        oof = rng.random(n)
        cal_oof = np.clip(oof + rng.normal(0, 0.01, n), 0, 1)

        outer_splits = [
            (
                np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.intp),
                np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19], dtype=np.intp),
            ),
            (
                np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19], dtype=np.intp),
                np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.intp),
            ),
        ]

        if_pred = [rng.random(10), rng.random(10)]

        from lizyml.core.types.artifacts import (
            DataFingerprint,
            SplitIndices,
        )

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

        y = pd.Series((oof > 0.5).astype(int))

        fr = FitResult(
            oof_pred=oof,
            if_pred_per_fold=if_pred,
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

        return fr, y

    def test_calibrated_has_oof_per_fold(self) -> None:
        """metrics['calibrated'] must contain 'oof_per_fold' key."""
        fr, y = self._make_fit_result_with_calibrator()
        evaluator = Evaluator(task="binary")
        raw_metrics = evaluator.evaluate(fr, y, ["auc", "logloss"])

        result = assemble_calibrated_metrics(
            fr, y, ["auc", "logloss"], evaluator, raw_metrics
        )

        assert "calibrated" in result
        assert "oof" in result["calibrated"]
        assert "oof_per_fold" in result["calibrated"], (
            "calibrated branch is missing 'oof_per_fold' key"
        )
        assert isinstance(result["calibrated"]["oof_per_fold"], list)
        assert len(result["calibrated"]["oof_per_fold"]) == 2

    def test_calibrated_excludes_if_metrics(self) -> None:
        """calibrated branch must NOT contain IF metrics (leakage)."""
        fr, y = self._make_fit_result_with_calibrator()
        evaluator = Evaluator(task="binary")
        raw_metrics = evaluator.evaluate(fr, y, ["auc"])

        result = assemble_calibrated_metrics(fr, y, ["auc"], evaluator, raw_metrics)

        assert "if_mean" not in result.get("calibrated", {})
        assert "if_per_fold" not in result.get("calibrated", {})
