"""Tests for IsotonicCalibrator using LGBM monotone constraints (BLUEPRINT §12.2).

Covers:
- fit/predict with synthetic data
- Output range [0, 1]
- Monotonicity of predictions
- Booster API usage (lgb.train, not LGBMRegressor)
- Early stopping with internal validation split
- Reproducibility with seed
- Custom params via calibration.params
- Small sample robustness (< 20 rows, early stopping disabled)
- E2E via Model.fit with calibration.method: "isotonic"
"""

from __future__ import annotations

import numpy as np
import pytest

from lizyml import Model
from lizyml.calibration.isotonic import IsotonicCalibrator
from tests._helpers import make_binary_df


def _synthetic_scores(n: int = 500, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n).astype(float)
    scores = y * 2.0 - 1.0 + rng.normal(0, 0.5, n)
    return scores, y


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


class TestIsotonicCalibrator:
    def test_fit_predict(self) -> None:
        scores, y = _synthetic_scores()
        cal = IsotonicCalibrator()
        cal.fit(scores, y)
        pred = cal.predict(scores)
        assert pred.shape == scores.shape

    def test_output_range(self) -> None:
        scores, y = _synthetic_scores()
        cal = IsotonicCalibrator()
        cal.fit(scores, y)
        pred = cal.predict(scores)
        assert pred.min() >= 0.0
        assert pred.max() <= 1.0

    def test_monotone(self) -> None:
        """Predictions should be monotonically non-decreasing for sorted inputs."""
        scores, y = _synthetic_scores()
        cal = IsotonicCalibrator()
        cal.fit(scores, y)

        sorted_scores = np.sort(scores)
        pred = cal.predict(sorted_scores)
        # Allow tiny numerical tolerance
        diffs = np.diff(pred)
        assert np.all(diffs >= -1e-10), f"Non-monotone diffs: {diffs[diffs < -1e-10]}"

    def test_name(self) -> None:
        cal = IsotonicCalibrator()
        assert cal.name == "isotonic"

    def test_custom_params(self) -> None:
        """User can override defaults (except monotone_constraints) via params."""
        scores, y = _synthetic_scores()
        cal = IsotonicCalibrator(
            params={"num_boost_round": 50, "max_depth": 2, "learning_rate": 0.1}
        )
        cal.fit(scores, y)
        pred = cal.predict(scores)
        assert pred.shape == scores.shape
        assert pred.min() >= 0.0
        assert pred.max() <= 1.0

    def test_monotone_constraint_enforced(self) -> None:
        """User cannot override monotone_constraints."""
        scores, y = _synthetic_scores()
        cal = IsotonicCalibrator(params={"monotone_constraints": [-1]})
        cal.fit(scores, y)
        # monotone_constraints should still be [1]
        sorted_scores = np.sort(scores)
        pred = cal.predict(sorted_scores)
        diffs = np.diff(pred)
        assert np.all(diffs >= -1e-10)

    def test_uses_booster_api(self) -> None:
        """After fit, _model should be a lgb.Booster, not LGBMRegressor."""
        import lightgbm as lgbm

        scores, y = _synthetic_scores()
        cal = IsotonicCalibrator()
        cal.fit(scores, y)
        assert isinstance(cal._model, lgbm.Booster)

    def test_early_stopping_configured(self) -> None:
        """With n >= 20, early stopping should be configured (iteration <= 1000)."""
        scores, y = _synthetic_scores(n=500)
        cal = IsotonicCalibrator()
        cal.fit(scores, y)
        assert cal._model is not None
        # Model should have trained (at least some rounds)
        assert cal._model.current_iteration() >= 1
        # Should not exceed num_boost_round
        assert cal._model.current_iteration() <= 1000

    def test_reproducibility_with_seed(self) -> None:
        """Same seed should produce identical predictions."""
        scores, y = _synthetic_scores()
        cal1 = IsotonicCalibrator(params={"seed": 123})
        cal1.fit(scores, y)
        pred1 = cal1.predict(scores)

        cal2 = IsotonicCalibrator(params={"seed": 123})
        cal2.fit(scores, y)
        pred2 = cal2.predict(scores)

        np.testing.assert_array_equal(pred1, pred2)

    def test_small_sample_no_early_stopping(self) -> None:
        """With < 20 samples, early stopping should be disabled (no crash)."""
        scores, y = _synthetic_scores(n=15, seed=0)
        cal = IsotonicCalibrator()
        cal.fit(scores, y)
        pred = cal.predict(scores)
        assert pred.shape == scores.shape
        assert pred.min() >= 0.0
        assert pred.max() <= 1.0

    def test_custom_validation_ratio_and_seed(self) -> None:
        """User can override validation_ratio and seed via params."""
        scores, y = _synthetic_scores()
        cal = IsotonicCalibrator(params={"validation_ratio": 0.2, "seed": 99})
        cal.fit(scores, y)
        pred = cal.predict(scores)
        assert pred.shape == scores.shape
        assert pred.min() >= 0.0
        assert pred.max() <= 1.0

    def test_invalid_validation_ratio(self) -> None:
        """validation_ratio outside (0, 1) should raise CONFIG_INVALID."""
        from lizyml.core.exceptions import ErrorCode, LizyMLError

        with pytest.raises(LizyMLError) as exc_info:
            IsotonicCalibrator(params={"validation_ratio": 1.0})
        assert exc_info.value.code == ErrorCode.CONFIG_INVALID

        with pytest.raises(LizyMLError):
            IsotonicCalibrator(params={"validation_ratio": 0.0})

    def test_output_range_not_compressed(self) -> None:
        """Calibrated outputs must span a wide range, not compressed.

        With well-separated classes (logits ~ -1 for y=0, +1 for y=1), the
        calibrator should produce probabilities both below 0.3 and above 0.7.
        A double-sigmoid bug would compress everything to ~0.5-0.73.
        """
        scores, y = _synthetic_scores(n=1000, seed=42)
        cal = IsotonicCalibrator()
        cal.fit(scores, y)
        pred = cal.predict(scores)
        assert pred.min() < 0.3, f"min={pred.min():.4f}, expected < 0.3 (compressed?)"
        assert pred.max() > 0.7, f"max={pred.max():.4f}, expected > 0.7 (compressed?)"

    def test_extreme_scores_produce_extreme_probabilities(self) -> None:
        """Extreme logits (e.g. +/-10) should yield near-extreme probabilities.

        With the double-sigmoid bug, sigmoid(sigmoid(10)) ~ 0.73, which would
        fail the > 0.9 assertion.
        """
        scores, y = _synthetic_scores(n=500, seed=42)
        cal = IsotonicCalibrator()
        cal.fit(scores, y)
        extreme = np.array([-10.0, 10.0])
        pred = cal.predict(extreme)
        assert pred[0] < 0.1, f"pred(-10)={pred[0]:.4f}, expected < 0.1"
        assert pred[1] > 0.9, f"pred(+10)={pred[1]:.4f}, expected > 0.9"

    def test_params_not_mutated(self) -> None:
        """Constructor should not mutate the caller's params dict."""
        params = {"validation_ratio": 0.2, "seed": 99, "max_depth": 4}
        original_keys = set(params.keys())
        IsotonicCalibrator(params=params)
        assert set(params.keys()) == original_keys


# ---------------------------------------------------------------------------
# E2E tests
# ---------------------------------------------------------------------------


class TestIsotonicE2E:
    def test_model_fit_with_isotonic(self) -> None:
        df = make_binary_df(n=300)
        cfg = {
            "config_version": 1,
            "task": "binary",
            "data": {"target": "target"},
            "split": {
                "method": "stratified_kfold",
                "n_splits": 3,
                "random_state": 42,
            },
            "model": {"name": "lgbm", "params": {"n_estimators": 20}},
            "training": {"seed": 0},
            "calibration": {"method": "isotonic", "n_splits": 3},
        }
        model = Model(cfg, data=df)
        fr = model.fit()
        assert fr.calibrator is not None
        assert fr.calibrator.method == "isotonic"
        assert fr.calibrator.calibrated_oof.min() >= 0.0
        assert fr.calibrator.calibrated_oof.max() <= 1.0

    def test_model_fit_with_isotonic_custom_params(self) -> None:
        df = make_binary_df(n=300)
        cfg = {
            "config_version": 1,
            "task": "binary",
            "data": {"target": "target"},
            "split": {
                "method": "stratified_kfold",
                "n_splits": 3,
                "random_state": 42,
            },
            "model": {"name": "lgbm", "params": {"n_estimators": 20}},
            "training": {"seed": 0},
            "calibration": {
                "method": "isotonic",
                "n_splits": 3,
                "params": {"num_boost_round": 50},
            },
        }
        model = Model(cfg, data=df)
        fr = model.fit()
        assert fr.calibrator is not None
