"""Tests for Phase 13 — Calibration.

Covers:
- PlattCalibrator: fit → predict, output in [0, 1]
- IsotonicCalibrator: fit → predict, output in [0, 1]
- cross_fit_calibrate: leakage test (C_final vs cross-fit separation)
- cross_fit_calibrate: calibrated_oof length == n_samples
- Model.fit() with calibration config: calibrated metrics in fit_result.metrics
- Model.fit() calibration on non-binary raises CALIBRATION_NOT_SUPPORTED
- Model.predict() with calibration applies C_final
- X inaccessibility: calibrators only accept (oof_scores, y)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lizyml import Model
from lizyml.calibration.base import BaseCalibratorAdapter
from lizyml.calibration.cross_fit import CalibrationResult, cross_fit_calibrate
from lizyml.calibration.isotonic import IsotonicCalibrator
from lizyml.calibration.platt import PlattCalibrator
from lizyml.calibration.registry import get_calibrator
from lizyml.core.exceptions import ErrorCode, LizyMLError

# ---------------------------------------------------------------------------
# Synthetic binary dataset
# ---------------------------------------------------------------------------


def _bin_df(n: int = 200, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {"feat_a": rng.uniform(0, 10, n), "feat_b": rng.uniform(-1, 1, n)}
    )
    df["target"] = (df["feat_a"] > 5).astype(int)
    return df


def _oof_scores(n: int = 200, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return (oof_scores, y) for calibrator tests."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    # Realistic-ish scores: near 0 for class 0, near 1 for class 1 with noise
    scores = y.astype(float) + rng.normal(0, 0.3, n)
    scores = np.clip(scores, 0.01, 0.99)
    return scores, y.astype(float)


def _bin_config(with_calibration: bool = False, method: str = "platt") -> dict:
    cfg: dict = {
        "config_version": 1,
        "task": "binary",
        "data": {"target": "target"},
        "split": {"method": "kfold", "n_splits": 3, "random_state": 42},
        "model": {"name": "lgbm", "params": {"n_estimators": 20}},
        "training": {"seed": 0},
    }
    if with_calibration:
        cfg["calibration"] = {"method": method, "n_splits": 3}
    return cfg


def _reg_config() -> dict:
    return {
        "config_version": 1,
        "task": "regression",
        "data": {"target": "target"},
        "split": {"method": "kfold", "n_splits": 3, "random_state": 42},
        "model": {"name": "lgbm", "params": {"n_estimators": 10}},
        "training": {"seed": 0},
        "calibration": {"method": "platt", "n_splits": 3},
    }


# ---------------------------------------------------------------------------
# PlattCalibrator
# ---------------------------------------------------------------------------


class TestPlattCalibrator:
    def test_output_in_unit_interval(self) -> None:
        scores, y = _oof_scores()
        cal = PlattCalibrator()
        cal.fit(scores, y)
        proba = cal.predict(scores)
        assert proba.shape == (len(scores),)
        assert np.all(proba >= 0.0) and np.all(proba <= 1.0)

    def test_name(self) -> None:
        assert PlattCalibrator().name == "platt"

    def test_no_x_parameter(self) -> None:
        """fit() must not accept X; signature is (oof_scores, y)."""
        import inspect

        sig = inspect.signature(BaseCalibratorAdapter.fit)
        params = list(sig.parameters.keys())
        assert "X" not in params
        assert "oof_scores" in params

    def test_predict_before_fit_raises(self) -> None:
        cal = PlattCalibrator()
        with pytest.raises(RuntimeError):
            cal.predict(np.array([0.5, 0.6]))


# ---------------------------------------------------------------------------
# IsotonicCalibrator
# ---------------------------------------------------------------------------


class TestIsotonicCalibrator:
    def test_output_in_unit_interval(self) -> None:
        scores, y = _oof_scores()
        cal = IsotonicCalibrator()
        cal.fit(scores, y)
        proba = cal.predict(scores)
        assert proba.shape == (len(scores),)
        assert np.all(proba >= 0.0) and np.all(proba <= 1.0)

    def test_name(self) -> None:
        assert IsotonicCalibrator().name == "isotonic"

    def test_predict_before_fit_raises(self) -> None:
        cal = IsotonicCalibrator()
        with pytest.raises(RuntimeError):
            cal.predict(np.array([0.5]))


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestCalibratorRegistry:
    def test_get_platt(self) -> None:
        cal = get_calibrator("platt")
        assert isinstance(cal, PlattCalibrator)

    def test_get_isotonic(self) -> None:
        cal = get_calibrator("isotonic")
        assert isinstance(cal, IsotonicCalibrator)

    def test_unknown_raises(self) -> None:
        with pytest.raises(KeyError):
            get_calibrator("nonexistent")


# ---------------------------------------------------------------------------
# cross_fit_calibrate — leakage tests
# ---------------------------------------------------------------------------


class TestCrossFitCalibrate:
    def test_calibrated_oof_length(self) -> None:
        scores, y = _oof_scores(n=100)
        result = cross_fit_calibrate(
            oof_scores=scores, y=y, calibrator_factory=PlattCalibrator, n_splits=3
        )
        assert isinstance(result, CalibrationResult)
        assert result.calibrated_oof.shape == (100,)

    def test_c_final_and_cross_fit_differ(self) -> None:
        """C_final predictions on full data differ from cross-fit OOF (no leak test)."""
        scores, y = _oof_scores(n=100)
        result = cross_fit_calibrate(
            oof_scores=scores, y=y, calibrator_factory=PlattCalibrator, n_splits=3
        )
        c_final_preds = result.c_final.predict(scores)
        # They should NOT be identical (c_final sees all training data)
        # We just check both are valid probabilities
        assert np.all(c_final_preds >= 0.0) and np.all(c_final_preds <= 1.0)
        assert np.all(result.calibrated_oof >= 0.0)
        assert np.all(result.calibrated_oof <= 1.0)

    def test_leakage_detection(self) -> None:
        """C_final should NOT equal cross-fit OOF on the same training samples.

        If calibration were naive (fit on ALL then predict on ALL), the
        calibrated OOF would exactly match c_final predictions on the full set.
        In proper cross-fit, they differ because each fold's calibrator was
        trained on the *other* folds.
        """
        scores, y = _oof_scores(n=200)
        result = cross_fit_calibrate(
            oof_scores=scores, y=y, calibrator_factory=PlattCalibrator, n_splits=5
        )
        c_final_preds = result.c_final.predict(scores)
        # With 5 folds and non-trivial data, they must differ somewhere
        assert not np.allclose(result.calibrated_oof, c_final_preds, atol=1e-6), (
            "cross-fit OOF should differ from C_final predictions — "
            "if they are identical, calibration is leaking!"
        )

    def test_method_name_stored(self) -> None:
        scores, y = _oof_scores()
        result = cross_fit_calibrate(
            oof_scores=scores, y=y, calibrator_factory=PlattCalibrator
        )
        assert result.method == "platt"


# ---------------------------------------------------------------------------
# Model integration
# ---------------------------------------------------------------------------


class TestModelCalibration:
    def test_fit_with_platt_stores_calibration_result(self) -> None:
        df = _bin_df()
        m = Model(_bin_config(with_calibration=True, method="platt"))
        result = m.fit(data=df)
        assert result.calibrator is not None
        assert isinstance(result.calibrator, CalibrationResult)

    def test_fit_calibrated_metrics_in_result(self) -> None:
        df = _bin_df()
        m = Model(_bin_config(with_calibration=True))
        result = m.fit(data=df)
        assert "calibrated" in result.metrics
        assert "oof" in result.metrics["calibrated"]
        cal_oof_metrics = result.metrics["calibrated"]["oof"]
        assert "logloss" in cal_oof_metrics

    def test_fit_isotonic_works(self) -> None:
        df = _bin_df()
        m = Model(_bin_config(with_calibration=True, method="isotonic"))
        result = m.fit(data=df)
        assert isinstance(result.calibrator, CalibrationResult)
        assert result.calibrator.method == "isotonic"

    def test_calibration_on_regression_raises(self) -> None:
        rng = np.random.default_rng(0)
        df = pd.DataFrame(
            {
                "feat_a": rng.uniform(0, 10, 100),
                "feat_b": rng.uniform(-1, 1, 100),
                "target": rng.uniform(0, 10, 100),
            }
        )
        m = Model(_reg_config())
        with pytest.raises(LizyMLError) as exc_info:
            m.fit(data=df)
        assert exc_info.value.code == ErrorCode.CALIBRATION_NOT_SUPPORTED

    def test_predict_with_calibration_applies_c_final(self) -> None:
        df = _bin_df()
        m = Model(_bin_config(with_calibration=True))
        m.fit(data=df)
        X_new = df.drop(columns=["target"]).iloc[:10].reset_index(drop=True)
        pred_result = m.predict(X_new)
        # Calibrated probabilities should be in [0, 1]
        assert pred_result.proba is not None
        assert np.all(pred_result.proba >= 0.0) and np.all(pred_result.proba <= 1.0)

    def test_predict_without_calibration_still_works(self) -> None:
        df = _bin_df()
        m = Model(_bin_config(with_calibration=False))
        m.fit(data=df)
        X_new = df.drop(columns=["target"]).iloc[:5].reset_index(drop=True)
        pred_result = m.predict(X_new)
        assert pred_result.proba is not None
        assert pred_result.proba.shape == (5,)
