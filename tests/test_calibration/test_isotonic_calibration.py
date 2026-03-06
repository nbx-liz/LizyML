"""Tests for IsotonicCalibrator using LGBM monotone constraints (BLUEPRINT §12.2).

Covers:
- fit/predict with synthetic data
- Output range [0, 1]
- Monotonicity of predictions
- E2E via Model.fit with calibration.method: "isotonic"
- Custom params via calibration.params
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from lizyml import Model
from lizyml.calibration.isotonic import IsotonicCalibrator

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bin_df(n: int = 300, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {"feat_a": rng.uniform(0, 10, n), "feat_b": rng.uniform(-1, 1, n)}
    )
    df["target"] = (df["feat_a"] > 5).astype(int)
    return df


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
        scores, y = _synthetic_scores()
        cal = IsotonicCalibrator(params={"n_estimators": 50, "max_depth": 2})
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


# ---------------------------------------------------------------------------
# E2E tests
# ---------------------------------------------------------------------------


class TestIsotonicE2E:
    def test_model_fit_with_isotonic(self) -> None:
        df = _bin_df()
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
        df = _bin_df()
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
                "params": {"n_estimators": 50},
            },
        }
        model = Model(cfg, data=df)
        fr = model.fit()
        assert fr.calibrator is not None
