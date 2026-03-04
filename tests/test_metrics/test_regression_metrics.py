"""Tests for MAPE and HuberLoss regression metrics (H-0004)."""

from __future__ import annotations

import numpy as np
import pytest

from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.metrics import MAPE, HuberLoss, get_metric, get_metrics_for_task


# ---------------------------------------------------------------------------
# MAPE
# ---------------------------------------------------------------------------


class TestMAPE:
    def test_correctness(self) -> None:
        y_true = np.array([100.0, 200.0, 300.0])
        y_pred = np.array([110.0, 190.0, 300.0])
        # |errors| / y_true = [0.10, 0.05, 0.00] → mean = 0.05 → 5.0 %
        assert MAPE()(y_true, y_pred) == pytest.approx(5.0)

    def test_perfect_prediction_is_zero(self) -> None:
        y = np.array([1.0, 2.0, 3.0])
        assert MAPE()(y, y) == pytest.approx(0.0)

    def test_zero_in_y_true_raises(self) -> None:
        y_true = np.array([0.0, 1.0, 2.0])
        y_pred = np.array([0.0, 1.0, 2.0])
        with pytest.raises(LizyMLError) as exc:
            MAPE()(y_true, y_pred)
        assert exc.value.code == ErrorCode.UNSUPPORTED_METRIC

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(LizyMLError) as exc:
            MAPE()(np.array([1.0, 2.0]), np.array([1.0]))
        assert exc.value.code == ErrorCode.UNSUPPORTED_METRIC

    def test_contracts(self) -> None:
        m = MAPE()
        assert m.name == "mape"
        assert m.needs_proba is False
        assert m.greater_is_better is False

    def test_registry_lookup(self) -> None:
        m = get_metric("mape")
        assert isinstance(m, MAPE)

    def test_task_compat_regression(self) -> None:
        metrics = get_metrics_for_task(["mape"], "regression")
        assert len(metrics) == 1

    def test_task_compat_binary_raises(self) -> None:
        with pytest.raises(LizyMLError) as exc:
            get_metrics_for_task(["mape"], "binary")
        assert exc.value.code == ErrorCode.UNSUPPORTED_METRIC


# ---------------------------------------------------------------------------
# HuberLoss
# ---------------------------------------------------------------------------


class TestHuberLoss:
    def test_squared_region(self) -> None:
        # |error| = 0.5 <= delta=1.0 → 0.5 * 0.5^2 = 0.125
        y_true = np.array([1.0])
        y_pred = np.array([1.5])
        assert HuberLoss(delta=1.0)(y_true, y_pred) == pytest.approx(0.125)

    def test_linear_region(self) -> None:
        # |error| = 2.0 > delta=1.0 → 1.0 * (2.0 - 0.5) = 1.5
        y_true = np.array([3.0])
        y_pred = np.array([1.0])
        assert HuberLoss(delta=1.0)(y_true, y_pred) == pytest.approx(1.5)

    def test_perfect_prediction_is_zero(self) -> None:
        y = np.array([1.0, 2.0, 3.0])
        assert HuberLoss()(y, y) == pytest.approx(0.0)

    def test_boundary_at_delta(self) -> None:
        # |error| == delta=1.0 → squared: 0.5 * 1.0^2 = 0.5
        y_true = np.array([2.0])
        y_pred = np.array([1.0])
        assert HuberLoss(delta=1.0)(y_true, y_pred) == pytest.approx(0.5)

    def test_custom_delta(self) -> None:
        # |error|=0.5 with delta=2.0 → squared region → 0.5 * 0.25 = 0.125
        y_true = np.array([1.0])
        y_pred = np.array([1.5])
        assert HuberLoss(delta=2.0)(y_true, y_pred) == pytest.approx(0.125)

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(LizyMLError) as exc:
            HuberLoss()(np.array([1.0, 2.0]), np.array([1.0]))
        assert exc.value.code == ErrorCode.UNSUPPORTED_METRIC

    def test_contracts(self) -> None:
        m = HuberLoss()
        assert m.name == "huber"
        assert m.needs_proba is False
        assert m.greater_is_better is False

    def test_registry_lookup(self) -> None:
        m = get_metric("huber")
        assert isinstance(m, HuberLoss)

    def test_task_compat_regression(self) -> None:
        metrics = get_metrics_for_task(["huber"], "regression")
        assert len(metrics) == 1

    def test_task_compat_multiclass_raises(self) -> None:
        with pytest.raises(LizyMLError) as exc:
            get_metrics_for_task(["huber"], "multiclass")
        assert exc.value.code == ErrorCode.UNSUPPORTED_METRIC
