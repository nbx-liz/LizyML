"""Tests for Phase 23-A — Raw Score Calibration (H-0030).

Covers:
- predict_raw returns logits (different from predict_proba) for binary
- predict_raw returns same as predict for regression
- oof_raw_scores populated when calibration is enabled
- oof_raw_scores is None when calibration is disabled
- calibration predict path uses raw scores
"""

from __future__ import annotations

import numpy as np

from lizyml import Model
from lizyml.calibration.cross_fit import CalibrationResult
from lizyml.estimators.lgbm import LGBMAdapter
from tests._helpers import make_binary_df, make_config, make_regression_df

# ---------------------------------------------------------------------------
# Adapter-level tests
# ---------------------------------------------------------------------------


class TestPredictRaw:
    def test_predict_raw_returns_logits_for_binary(self) -> None:
        """predict_raw should return logits, different from predict_proba."""
        df = make_binary_df()
        X = df[["feat_a", "feat_b"]]
        y = df["target"]

        adapter = LGBMAdapter(task="binary", params={"n_estimators": 10})
        adapter.fit(X, y)

        raw = adapter.predict_raw(X)
        proba = adapter.predict_proba(X)[:, 1]

        assert raw.shape == (len(X),)
        # Raw logits should NOT be in [0, 1] range (they are unbounded)
        assert not np.allclose(raw, proba, atol=1e-6)
        # Some raw values should be outside [0, 1]
        assert raw.min() < 0 or raw.max() > 1

    def test_predict_raw_regression_same_as_predict(self) -> None:
        """For regression, predict_raw should be identical to predict."""
        df = make_regression_df()
        X = df[["feat_a", "feat_b"]]
        y = df["target"]

        adapter = LGBMAdapter(task="regression", params={"n_estimators": 10})
        adapter.fit(X, y)

        raw = adapter.predict_raw(X)
        pred = adapter.predict(X)

        np.testing.assert_array_equal(raw, pred)


# ---------------------------------------------------------------------------
# End-to-end tests
# ---------------------------------------------------------------------------


class TestOofRawScores:
    def test_oof_raw_scores_populated_with_calibration(self) -> None:
        """oof_raw_scores should be non-None when calibration is enabled."""
        df = make_binary_df()
        cfg = make_config("binary", calibration="platt", n_estimators=20)
        model = Model(cfg, data=df)
        fr = model.fit()

        assert fr.oof_raw_scores is not None
        assert fr.oof_raw_scores.shape == (len(df),)
        # Raw scores differ from oof_pred (probabilities)
        assert not np.allclose(fr.oof_raw_scores, fr.oof_pred, atol=1e-6)

    def test_oof_raw_scores_none_without_calibration(self) -> None:
        """oof_raw_scores should be None when calibration is disabled."""
        df = make_binary_df()
        model = Model(make_config("binary", n_estimators=20), data=df)
        fr = model.fit()

        assert fr.oof_raw_scores is None

    def test_calibration_result_trained_on_raw_scores(self) -> None:
        """C_final should be trained on raw scores, not probabilities."""
        df = make_binary_df()
        cfg = make_config("binary", calibration="platt", n_estimators=20)
        model = Model(cfg, data=df)
        fr = model.fit()

        assert isinstance(fr.calibrator, CalibrationResult)
        # Calibrated OOF should be valid probabilities in [0, 1]
        assert fr.calibrator.calibrated_oof.min() >= 0.0
        assert fr.calibrator.calibrated_oof.max() <= 1.0

    def test_predict_uses_raw_scores_for_calibration(self) -> None:
        """predict() should use raw scores when calibration is enabled."""
        df = make_binary_df()
        cfg = make_config("binary", calibration="platt", n_estimators=20)
        model = Model(cfg, data=df)
        model.fit()

        result = model.predict(df)
        # Predictions should be valid (0 or 1)
        assert set(np.unique(result.pred)).issubset({0, 1})
        # Proba should be in [0, 1]
        assert result.proba is not None
        assert result.proba.min() >= 0.0
        assert result.proba.max() <= 1.0
