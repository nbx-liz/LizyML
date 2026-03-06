"""Tests for multiclass OvR metrics: AUC, AUC-PR, Brier (H-0018)."""

from __future__ import annotations

import numpy as np
import pytest

from lizyml.core.exceptions import LizyMLError
from lizyml.metrics.classification import AUC, AUCPR, Brier
from lizyml.metrics.registry import get_metrics_for_task


@pytest.fixture()
def multiclass_data() -> tuple[np.ndarray, np.ndarray]:
    """3-class multiclass data with reasonable predictions."""
    rng = np.random.default_rng(42)
    n = 100
    y_true = np.array([0] * 40 + [1] * 30 + [2] * 30)
    # Create decent predictions: high prob for correct class
    y_pred = rng.dirichlet([0.3, 0.3, 0.3], size=n)
    for i in range(n):
        y_pred[i, y_true[i]] += 1.0
    # Re-normalise
    y_pred = y_pred / y_pred.sum(axis=1, keepdims=True)
    return y_true, y_pred


class TestAUCMulticlassOvR:
    def test_returns_float_in_range(
        self, multiclass_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        y_true, y_pred = multiclass_data
        metric = AUC()
        result = metric(y_true, y_pred)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_good_predictions_high_auc(
        self, multiclass_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        y_true, y_pred = multiclass_data
        metric = AUC()
        result = metric(y_true, y_pred)
        assert result > 0.8

    def test_binary_unchanged(self) -> None:
        y_true = np.array([0, 0, 1, 1, 1])
        y_pred = np.array([0.1, 0.2, 0.8, 0.9, 0.7])
        metric = AUC()
        result = metric(y_true, y_pred)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0


class TestAUCPRMulticlassOvR:
    def test_returns_float_in_range(
        self, multiclass_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        y_true, y_pred = multiclass_data
        metric = AUCPR()
        result = metric(y_true, y_pred)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_binary_unchanged(self) -> None:
        y_true = np.array([0, 0, 1, 1, 1])
        y_pred = np.array([0.1, 0.2, 0.8, 0.9, 0.7])
        metric = AUCPR()
        result = metric(y_true, y_pred)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0


class TestBrierMulticlassOvR:
    def test_returns_nonnegative_float(
        self, multiclass_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        y_true, y_pred = multiclass_data
        metric = Brier()
        result = metric(y_true, y_pred)
        assert isinstance(result, float)
        assert result >= 0.0

    def test_good_predictions_low_brier(
        self, multiclass_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        y_true, y_pred = multiclass_data
        metric = Brier()
        result = metric(y_true, y_pred)
        assert result < 0.3

    def test_binary_unchanged(self) -> None:
        y_true = np.array([0, 0, 1, 1, 1])
        y_pred = np.array([0.1, 0.2, 0.8, 0.9, 0.7])
        metric = Brier()
        result = metric(y_true, y_pred)
        assert isinstance(result, float)
        assert result >= 0.0


class TestMulticlassRegistered:
    def test_auc_registered_for_multiclass(self) -> None:
        metrics = get_metrics_for_task(["auc"], "multiclass")
        assert len(metrics) == 1
        assert metrics[0].name == "auc"

    def test_auc_pr_registered_for_multiclass(self) -> None:
        metrics = get_metrics_for_task(["auc_pr"], "multiclass")
        assert len(metrics) == 1
        assert metrics[0].name == "auc_pr"

    def test_brier_registered_for_multiclass(self) -> None:
        metrics = get_metrics_for_task(["brier"], "multiclass")
        assert len(metrics) == 1
        assert metrics[0].name == "brier"

    def test_regression_excluded(self) -> None:
        with pytest.raises(LizyMLError) as exc_info:
            get_metrics_for_task(["auc"], "regression")
        assert exc_info.value.code.value == "UNSUPPORTED_METRIC"
