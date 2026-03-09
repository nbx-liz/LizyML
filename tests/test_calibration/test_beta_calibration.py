"""Tests for Phase 23-B — Beta Calibration (H-0031).

Covers:
- BetaCalibrator fit/predict
- Output range [0, 1]
- E2E Model.fit with method="beta"
- Cross-fit leakage detection
- scipy missing error
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from lizyml import Model
from lizyml.calibration.beta import BetaCalibrator
from lizyml.calibration.cross_fit import CalibrationResult, cross_fit_calibrate
from lizyml.calibration.registry import get_calibrator
from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.splitters.kfold import KFoldSplitter
from tests._helpers import make_binary_df, make_config


def _oof_logits(n: int = 300, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return (logits, y) for calibrator tests."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n).astype(float)
    logits = y * 2.0 - 1.0 + rng.normal(0, 0.5, n)
    return logits, y


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


class TestBetaCalibrator:
    def test_fit_predict(self) -> None:
        logits, y = _oof_logits()
        cal = BetaCalibrator()
        cal.fit(logits, y)
        result = cal.predict(logits)

        assert result.shape == logits.shape
        assert result.dtype == np.float64

    def test_output_range(self) -> None:
        logits, y = _oof_logits()
        cal = BetaCalibrator()
        cal.fit(logits, y)
        result = cal.predict(logits)

        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_name(self) -> None:
        assert BetaCalibrator().name == "beta"

    def test_predict_before_fit_raises(self) -> None:
        cal = BetaCalibrator()
        with pytest.raises(RuntimeError, match="not been fitted"):
            cal.predict(np.array([0.0, 1.0]))

    def test_scipy_missing(self) -> None:
        logits, y = _oof_logits()
        cal = BetaCalibrator()
        with patch("lizyml.calibration.beta._scipy", None):
            with pytest.raises(LizyMLError) as exc_info:
                cal.fit(logits, y)
            assert exc_info.value.code == ErrorCode.OPTIONAL_DEP_MISSING


class TestBetaRegistry:
    def test_get_calibrator_beta(self) -> None:
        cal = get_calibrator("beta")
        assert isinstance(cal, BetaCalibrator)


class TestBetaCrossFit:
    def test_cross_fit_leakage(self) -> None:
        logits, y = _oof_logits()
        splits = list(
            KFoldSplitter(n_splits=5, shuffle=True, random_state=42).split(len(logits))
        )
        result = cross_fit_calibrate(
            oof_scores=logits,
            y=y,
            calibrator_factory=lambda: BetaCalibrator(),
            split_indices=splits,
        )
        assert isinstance(result, CalibrationResult)
        assert result.calibrated_oof.shape == logits.shape
        # Cross-fit OOF should differ from C_final predictions
        c_final_preds = result.c_final.predict(logits)
        assert not np.allclose(result.calibrated_oof, c_final_preds, atol=1e-6)


# ---------------------------------------------------------------------------
# E2E test
# ---------------------------------------------------------------------------


class TestBetaE2E:
    def test_model_fit_with_beta_calibration(self) -> None:
        df = make_binary_df()
        cfg = make_config("binary", calibration="beta", n_estimators=20)
        model = Model(cfg, data=df)
        fr = model.fit()

        assert isinstance(fr.calibrator, CalibrationResult)
        assert fr.calibrator.method == "beta"
        assert fr.calibrator.calibrated_oof.min() >= 0.0
        assert fr.calibrator.calibrated_oof.max() <= 1.0
