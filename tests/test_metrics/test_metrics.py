"""Tests for Phase 7 — Metrics.

Covers:
- Numerical correctness of each metric
- needs_proba / greater_is_better contracts
- Shape mismatch raises LizyMLError with UNSUPPORTED_METRIC
- Registry lookup by name
- Task-compatibility validation
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.metrics import (
    AUC,
    AUCPR,
    ECE,
    F1,
    MAE,
    R2,
    RMSE,
    RMSLE,
    Accuracy,
    Brier,
    LogLoss,
    get_metric,
    get_metrics_for_task,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def reg_data() -> tuple[np.ndarray, np.ndarray]:
    """Perfect predictions plus a slight error."""
    rng = np.random.default_rng(0)
    y_true = rng.uniform(0, 10, size=100)
    y_pred = y_true + rng.normal(0, 0.1, size=100)
    return y_true, y_pred


@pytest.fixture()
def clf_data() -> tuple[np.ndarray, np.ndarray]:
    """Binary labels with predicted probabilities."""
    rng = np.random.default_rng(1)
    y_true = rng.integers(0, 2, size=200).astype(float)
    # Good-ish probabilities: correlated with y_true
    noise = rng.uniform(0, 0.3, size=200)
    y_pred = np.clip(y_true * 0.8 + noise * 0.2, 0.05, 0.95)
    return y_true, y_pred


# ---------------------------------------------------------------------------
# Regression metrics — numerical correctness
# ---------------------------------------------------------------------------


class TestRMSE:
    def test_perfect(self) -> None:
        y = np.array([1.0, 2.0, 3.0])
        assert RMSE()(y, y) == pytest.approx(0.0)

    def test_known_value(self) -> None:
        y_true = np.array([0.0, 0.0, 0.0])
        y_pred = np.array([1.0, 1.0, 1.0])
        assert RMSE()(y_true, y_pred) == pytest.approx(1.0)

    def test_properties(self) -> None:
        m = RMSE()
        assert m.name == "rmse"
        assert m.needs_proba is False
        assert m.greater_is_better is False

    def test_shape_mismatch(self) -> None:
        with pytest.raises(LizyMLError) as exc_info:
            RMSE()(np.array([1.0, 2.0]), np.array([1.0]))
        assert exc_info.value.code == ErrorCode.UNSUPPORTED_METRIC


class TestMAE:
    def test_known_value(self) -> None:
        y_true = np.array([0.0, 1.0, 2.0])
        y_pred = np.array([1.0, 1.0, 1.0])
        assert MAE()(y_true, y_pred) == pytest.approx(2 / 3)

    def test_properties(self) -> None:
        m = MAE()
        assert m.name == "mae"
        assert m.needs_proba is False
        assert m.greater_is_better is False


class TestR2:
    def test_perfect(self) -> None:
        y = np.arange(10, dtype=float)
        assert R2()(y, y) == pytest.approx(1.0)

    def test_baseline(self) -> None:
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.full(3, 2.0)  # predicting the mean
        assert R2()(y_true, y_pred) == pytest.approx(0.0)

    def test_constant_target(self) -> None:
        y = np.ones(5)
        # Both perfect and imperfect predictions on constant target
        assert R2()(y, y) == pytest.approx(1.0)
        assert R2()(y, y + 1.0) == pytest.approx(0.0)

    def test_properties(self) -> None:
        m = R2()
        assert m.name == "r2"
        assert m.greater_is_better is True


class TestRMSLE:
    def test_perfect(self) -> None:
        y = np.array([1.0, 2.0, 3.0])
        assert RMSLE()(y, y) == pytest.approx(0.0)

    def test_known_value(self) -> None:
        y_true = np.array([0.0, 0.0])
        y_pred = np.array([1.0, 1.0])
        expected = math.log(2)  # log1p(1) - log1p(0)
        assert RMSLE()(y_true, y_pred) == pytest.approx(expected)

    def test_properties(self) -> None:
        m = RMSLE()
        assert m.name == "rmsle"
        assert m.greater_is_better is False


# ---------------------------------------------------------------------------
# Classification metrics — numerical correctness
# ---------------------------------------------------------------------------


class TestLogLoss:
    def test_near_perfect(self, clf_data: tuple[np.ndarray, np.ndarray]) -> None:
        y_true, y_pred = clf_data
        # LogLoss for a good predictor should be < 1
        assert LogLoss()(y_true, y_pred) < 1.0

    def test_properties(self) -> None:
        m = LogLoss()
        assert m.name == "logloss"
        assert m.needs_proba is True
        assert m.greater_is_better is False

    def test_length_mismatch(self) -> None:
        with pytest.raises(LizyMLError) as exc_info:
            LogLoss()(np.array([0.0, 1.0]), np.array([0.5]))
        assert exc_info.value.code == ErrorCode.UNSUPPORTED_METRIC


class TestAUC:
    def test_perfect(self) -> None:
        y_true = np.array([0.0, 0.0, 1.0, 1.0])
        y_pred = np.array([0.1, 0.2, 0.8, 0.9])
        assert AUC()(y_true, y_pred) == pytest.approx(1.0)

    def test_random(self) -> None:
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, 100).astype(float)
        y_pred = rng.uniform(0, 1, 100)
        # Random predictions should be ~0.5
        assert 0.3 < AUC()(y_true, y_pred) < 0.7

    def test_properties(self) -> None:
        m = AUC()
        assert m.name == "auc"
        assert m.needs_proba is True
        assert m.greater_is_better is True


class TestAUCPR:
    def test_perfect(self) -> None:
        y_true = np.array([0.0, 0.0, 1.0, 1.0])
        y_pred = np.array([0.1, 0.2, 0.8, 0.9])
        assert AUCPR()(y_true, y_pred) == pytest.approx(1.0)

    def test_properties(self) -> None:
        m = AUCPR()
        assert m.name == "auc_pr"
        assert m.needs_proba is True
        assert m.greater_is_better is True


class TestF1:
    def test_perfect(self) -> None:
        y_true = np.array([0, 0, 1, 1], dtype=float)
        y_pred = np.array([0, 0, 1, 1], dtype=float)
        assert F1()(y_true, y_pred) == pytest.approx(1.0)

    def test_with_proba(self) -> None:
        y_true = np.array([0, 0, 1, 1], dtype=float)
        y_pred = np.array([0.1, 0.2, 0.8, 0.9])
        # Threshold 0.5 → hard labels [0,0,1,1] → F1=1
        assert F1()(y_true, y_pred) == pytest.approx(1.0)

    def test_properties(self) -> None:
        m = F1()
        assert m.name == "f1"
        assert m.needs_proba is False
        assert m.greater_is_better is True


class TestAccuracy:
    def test_perfect(self) -> None:
        y_true = np.array([0, 1, 0, 1], dtype=float)
        assert Accuracy()(y_true, y_true) == pytest.approx(1.0)

    def test_half_correct(self) -> None:
        y_true = np.array([0, 0, 1, 1], dtype=float)
        y_pred = np.array([0, 1, 0, 1], dtype=float)
        assert Accuracy()(y_true, y_pred) == pytest.approx(0.5)

    def test_properties(self) -> None:
        m = Accuracy()
        assert m.name == "accuracy"
        assert m.needs_proba is False
        assert m.greater_is_better is True


class TestBrier:
    def test_perfect(self) -> None:
        y_true = np.array([0.0, 1.0])
        y_pred = np.array([0.0, 1.0])
        assert Brier()(y_true, y_pred) == pytest.approx(0.0)

    def test_worst(self) -> None:
        y_true = np.array([1.0, 0.0])
        y_pred = np.array([0.0, 1.0])
        assert Brier()(y_true, y_pred) == pytest.approx(1.0)

    def test_properties(self) -> None:
        m = Brier()
        assert m.name == "brier"
        assert m.needs_proba is True
        assert m.greater_is_better is False


class TestECE:
    def test_perfect_calibration(self) -> None:
        # Perfect calibration: prob=0.8 → accuracy=0.8
        n = 1000
        rng = np.random.default_rng(99)
        y_pred = np.full(n, 0.8)
        y_true = rng.binomial(1, 0.8, size=n).astype(float)
        ece = ECE(n_bins=10)(y_true, y_pred)
        # ECE should be small for well-calibrated predictions
        assert ece < 0.1

    def test_nonnegative(self, clf_data: tuple[np.ndarray, np.ndarray]) -> None:
        y_true, y_pred = clf_data
        assert ECE()(y_true, y_pred) >= 0.0

    def test_properties(self) -> None:
        m = ECE()
        assert m.name == "ece"
        assert m.needs_proba is True
        assert m.greater_is_better is False


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_get_known_metric(self) -> None:
        m = get_metric("rmse")
        assert isinstance(m, RMSE)

    def test_get_unknown_raises(self) -> None:
        with pytest.raises(LizyMLError) as exc_info:
            get_metric("nonexistent_metric")
        assert exc_info.value.code == ErrorCode.UNSUPPORTED_METRIC

    def test_get_metrics_for_task_regression(self) -> None:
        metrics = get_metrics_for_task(["rmse", "mae"], "regression")
        assert len(metrics) == 2
        assert metrics[0].name == "rmse"

    def test_get_metrics_for_task_binary(self) -> None:
        metrics = get_metrics_for_task(["auc", "logloss"], "binary")
        assert len(metrics) == 2

    def test_task_incompatible_metric_raises(self) -> None:
        with pytest.raises(LizyMLError) as exc_info:
            get_metrics_for_task(["rmse"], "binary")
        assert exc_info.value.code == ErrorCode.UNSUPPORTED_METRIC

    def test_regression_metric_not_in_binary(self) -> None:
        with pytest.raises(LizyMLError):
            get_metrics_for_task(["auc"], "regression")
