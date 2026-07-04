"""Regression tests for metric formulas (ECE, LogLoss, RMSLE).

Consolidated from the 2026-04 bugfix batches and the code-review fixes:
  BUG-1  ECE bin 'accuracy' must be fraction-of-positives, not binarized acc.
  ECE boundary: y_pred == 1.0 must land in the last bin.
  LogLoss must accept 2D multiclass y_pred.
  RMSLE must reject negative predictions/targets.
"""

from __future__ import annotations

import numpy as np
import pytest

from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.metrics.classification import ECE, LogLoss
from lizyml.metrics.regression import RMSLE


class TestECEFormula:
    """ECE 'accuracy' in each bin must be fraction-of-positives, not
    binarized-prediction accuracy."""

    def test_low_confidence_bin_well_calibrated(self) -> None:
        rng = np.random.default_rng(42)
        n = 2000
        y_pred = rng.uniform(0.1, 0.2, size=n)
        y_true = rng.binomial(1, y_pred).astype(float)
        ece = ECE(n_bins=10)(y_true, y_pred)
        assert ece < 0.05, (
            f"ECE={ece:.4f} is too high for well-calibrated low-confidence "
            "predictions (binarized accuracy instead of fraction-of-positives?)."
        )

    def test_high_confidence_bin_poorly_calibrated(self) -> None:
        n = 1000
        y_pred = np.full(n, 0.9)
        rng = np.random.default_rng(0)
        y_true = rng.binomial(1, 0.5, size=n).astype(float)
        ece = ECE(n_bins=10)(y_true, y_pred)
        assert 0.3 < ece < 0.5, f"ECE={ece:.4f}, expected ~0.4"

    def test_deterministic_hand_calculated(self) -> None:
        y_pred = np.array([0.2, 0.3, 0.7, 0.8])
        y_true = np.array([0.0, 0.0, 1.0, 1.0])
        ece = ECE(n_bins=1)(y_true, y_pred)
        assert ece == pytest.approx(0.0, abs=1e-10)


class TestECEBoundary:
    """ECE must include y_pred == 1.0 in the last bin."""

    def test_single_pred_one_wrong_class(self) -> None:
        y_true = np.array([0])
        y_pred = np.array([1.0])
        result = ECE(n_bins=10)(y_true, y_pred)
        assert result == pytest.approx(1.0, abs=1e-10)

    def test_all_ones_correct_class(self) -> None:
        y_true = np.array([1, 1, 1])
        y_pred = np.array([1.0, 1.0, 1.0])
        result = ECE(n_bins=10)(y_true, y_pred)
        assert result == pytest.approx(0.0, abs=1e-10)


class TestLogLossMulticlass:
    """LogLoss must handle 2D y_pred for multiclass (like AUC, Brier, AUCPR)."""

    def test_2d_returns_float(self) -> None:
        rng = np.random.default_rng(42)
        y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
        y_pred = rng.dirichlet([1, 1, 1], size=10)
        result = LogLoss()(y_true, y_pred)
        assert isinstance(result, float)
        assert result > 0.0

    def test_2d_good_predictions_lower_loss(self) -> None:
        n = 100
        y_true = np.array([0] * 40 + [1] * 30 + [2] * 30)
        rng = np.random.default_rng(42)
        y_good = rng.dirichlet([0.3, 0.3, 0.3], size=n)
        for i in range(n):
            y_good[i, y_true[i]] += 2.0
        y_good = y_good / y_good.sum(axis=1, keepdims=True)
        y_bad = rng.dirichlet([1, 1, 1], size=n)
        metric = LogLoss()
        assert metric(y_true, y_good) < metric(y_true, y_bad)

    def test_2d_shape_mismatch_raises(self) -> None:
        y_true = np.array([0, 1, 2])
        y_pred = np.random.default_rng(0).dirichlet([1, 1, 1], size=5)
        with pytest.raises(LizyMLError):
            LogLoss()(y_true, y_pred)

    def test_1d_binary_unchanged(self) -> None:
        y_true = np.array([0, 0, 1, 1, 1])
        y_pred = np.array([0.1, 0.2, 0.8, 0.9, 0.7])
        result = LogLoss()(y_true, y_pred)
        assert isinstance(result, float)
        assert result > 0.0


class TestRMSLENegativeGuard:
    """RMSLE must raise LizyMLError for negative predictions/targets."""

    def test_negative_pred_raises(self) -> None:
        with pytest.raises(LizyMLError) as exc_info:
            RMSLE()(np.array([1.0, 2.0, 3.0]), np.array([1.0, -0.5, 3.0]))
        assert exc_info.value.code == ErrorCode.UNSUPPORTED_METRIC

    def test_negative_true_raises(self) -> None:
        with pytest.raises(LizyMLError) as exc_info:
            RMSLE()(np.array([1.0, -1.0, 3.0]), np.array([1.0, 2.0, 3.0]))
        assert exc_info.value.code == ErrorCode.UNSUPPORTED_METRIC

    def test_nonneg_values_pass(self) -> None:
        result = RMSLE()(np.array([0.0, 1.0, 2.0]), np.array([0.1, 1.1, 1.9]))
        assert isinstance(result, float)
        assert result >= 0.0
