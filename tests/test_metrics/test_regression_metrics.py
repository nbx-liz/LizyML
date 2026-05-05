"""Tests for MAPE / HuberLoss / SMAPE / WAPE regression metrics.

H-0004: MAPE / HuberLoss
H-0071: SMAPE / WAPE (zero-tolerant percentage-style alternatives to MAPE)
"""

from __future__ import annotations

import numpy as np
import pytest

from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.metrics import (
    MAPE,
    SMAPE,
    WAPE,
    HuberLoss,
    get_metric,
    get_metrics_for_task,
)

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


# ---------------------------------------------------------------------------
# SMAPE (H-0071) — symmetric MAPE, range [0, 200], tolerates per-row zeros
# ---------------------------------------------------------------------------


class TestSMAPE:
    """Symmetric Mean Absolute Percentage Error.

    Formula::

        sMAPE = mean( 2 * |y_true - y_pred| / (|y_true| + |y_pred|) ) * 100

    Convention: when ``|y_true| + |y_pred| == 0`` for a row, that row's
    contribution is 0 (perfect prediction).
    """

    def test_correctness_handcomputed(self) -> None:
        # |yt - yp|         = [0,    1,   2]
        # |yt| + |yp|       = [200, 11,  10]
        # 2 * |yt-yp| / d   = [0,   2/11, 0.4]
        y_true = np.array([100.0, 5.0, 4.0])
        y_pred = np.array([100.0, 6.0, 6.0])
        expected = float(np.mean([0.0, 2 / 11, 0.4]) * 100)
        assert SMAPE()(y_true, y_pred) == pytest.approx(expected)

    def test_perfect_prediction_is_zero(self) -> None:
        y = np.array([1.0, 2.0, 3.0])
        assert SMAPE()(y, y) == pytest.approx(0.0)

    def test_zero_zero_row_treated_as_zero(self) -> None:
        # INV-2: |y_true| + |y_pred| == 0 → row contributes 0
        y_true = np.array([0.0, 1.0])
        y_pred = np.array([0.0, 1.0])
        assert SMAPE()(y_true, y_pred) == pytest.approx(0.0)

    def test_tolerates_zero_in_y_true(self) -> None:
        # MAPE would raise here; SMAPE must not.
        y_true = np.array([0.0, 5.0])
        y_pred = np.array([1.0, 4.0])
        # row 0: 2 * 1 / 1   = 2.0
        # row 1: 2 * 1 / 9   = 2/9
        expected = float(np.mean([2.0, 2 / 9]) * 100)
        assert SMAPE()(y_true, y_pred) == pytest.approx(expected)

    def test_tolerates_zero_in_y_pred(self) -> None:
        y_true = np.array([5.0, 4.0])
        y_pred = np.array([0.0, 4.0])
        # row 0: 2 * 5 / 5   = 2.0
        # row 1: 2 * 0 / 8   = 0.0
        expected = float(np.mean([2.0, 0.0]) * 100)
        assert SMAPE()(y_true, y_pred) == pytest.approx(expected)

    def test_upper_bound_200(self) -> None:
        # Opposite signs maximise sMAPE per row at 200.
        y_true = np.array([1.0, 1.0, 1.0])
        y_pred = np.array([-1.0, -1.0, -1.0])
        # 2 * |1 - (-1)| / (1 + 1) = 2.0 → 200%
        assert SMAPE()(y_true, y_pred) == pytest.approx(200.0)

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(LizyMLError) as exc:
            SMAPE()(np.array([1.0, 2.0]), np.array([1.0]))
        assert exc.value.code == ErrorCode.UNSUPPORTED_METRIC

    def test_contracts(self) -> None:
        m = SMAPE()
        assert m.name == "smape"
        assert m.needs_proba is False
        assert m.greater_is_better is False

    def test_registry_lookup(self) -> None:
        m = get_metric("smape")
        assert isinstance(m, SMAPE)

    def test_task_compat_regression(self) -> None:
        metrics = get_metrics_for_task(["smape"], "regression")
        assert len(metrics) == 1

    def test_task_compat_binary_raises(self) -> None:
        with pytest.raises(LizyMLError) as exc:
            get_metrics_for_task(["smape"], "binary")
        assert exc.value.code == ErrorCode.UNSUPPORTED_METRIC


# ---------------------------------------------------------------------------
# WAPE (H-0071) — Weighted Absolute Percentage Error
# ---------------------------------------------------------------------------


class TestWAPE:
    """Weighted Absolute Percentage Error.

    Formula::

        WAPE = sum(|y_true - y_pred|) / sum(|y_true|) * 100
             = MAE / mean(|y_true|) * 100

    Defined whenever ``sum(|y_true|) > 0`` (much weaker than MAPE's
    per-row guard).
    """

    def test_correctness_handcomputed(self) -> None:
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([11.0, 18.0, 33.0])
        # sum(|err|)   = 1 + 2 + 3 = 6
        # sum(|y_true|) = 60
        assert WAPE()(y_true, y_pred) == pytest.approx(10.0)

    def test_equivalent_to_mae_over_mean_abs_y_true(self) -> None:
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.5, 1.5, 3.5, 3.5])
        mae = float(np.mean(np.abs(y_true - y_pred)))
        mean_abs_yt = float(np.mean(np.abs(y_true)))
        expected = mae / mean_abs_yt * 100
        assert WAPE()(y_true, y_pred) == pytest.approx(expected)

    def test_perfect_prediction_is_zero(self) -> None:
        y = np.array([1.0, 2.0, 3.0])
        assert WAPE()(y, y) == pytest.approx(0.0)

    def test_partial_zero_in_y_true_does_not_raise(self) -> None:
        # Unlike MAPE, partial zeros are fine for WAPE.
        y_true = np.array([0.0, 1.0, 2.0])
        y_pred = np.array([0.5, 1.0, 2.0])
        # sum(|err|) = 0.5, sum(|y_true|) = 3.0 → 0.5 / 3.0 * 100
        expected = 0.5 / 3.0 * 100
        assert WAPE()(y_true, y_pred) == pytest.approx(expected)

    def test_all_zero_y_true_raises(self) -> None:
        # INV-3: only sum(|y_true|) == 0 raises.
        y_true = np.array([0.0, 0.0, 0.0])
        y_pred = np.array([0.0, 1.0, 2.0])
        with pytest.raises(LizyMLError) as exc:
            WAPE()(y_true, y_pred)
        assert exc.value.code == ErrorCode.UNSUPPORTED_METRIC

    def test_negative_y_true_uses_abs(self) -> None:
        # |y_true| in denominator means signs do not cancel.
        y_true = np.array([-10.0, 10.0])
        y_pred = np.array([-9.0, 9.0])
        # sum(|err|) = 1 + 1 = 2, sum(|y_true|) = 20 → 10%
        assert WAPE()(y_true, y_pred) == pytest.approx(10.0)

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(LizyMLError) as exc:
            WAPE()(np.array([1.0, 2.0]), np.array([1.0]))
        assert exc.value.code == ErrorCode.UNSUPPORTED_METRIC

    def test_contracts(self) -> None:
        m = WAPE()
        assert m.name == "wape"
        assert m.needs_proba is False
        assert m.greater_is_better is False

    def test_registry_lookup(self) -> None:
        m = get_metric("wape")
        assert isinstance(m, WAPE)

    def test_task_compat_regression(self) -> None:
        metrics = get_metrics_for_task(["wape"], "regression")
        assert len(metrics) == 1

    def test_task_compat_binary_raises(self) -> None:
        with pytest.raises(LizyMLError) as exc:
            get_metrics_for_task(["wape"], "binary")
        assert exc.value.code == ErrorCode.UNSUPPORTED_METRIC
