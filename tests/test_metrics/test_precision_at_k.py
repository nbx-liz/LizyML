"""Tests for Precision at K metric (H-0014)."""

from __future__ import annotations

import numpy as np
import pytest

from lizyml.core.exceptions import LizyMLError
from lizyml.metrics.classification import PrecisionAtK
from lizyml.metrics.registry import get_metrics_for_task


class TestPrecisionAtK:
    def test_perfect_ranking(self) -> None:
        """All positives ranked above all negatives → precision=1.0."""
        y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
        y_pred = np.array([0.9, 0.8, 0.7, 0.1, 0.05, 0.04, 0.03, 0.02, 0.01, 0.0])
        metric = PrecisionAtK(k=30)  # top 30% = 3 samples
        result = metric(y_true, y_pred)
        assert result == 1.0

    def test_worst_ranking(self) -> None:
        """All positives ranked below all negatives → precision=0.0."""
        y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
        y_pred = np.array([0.0, 0.01, 0.02, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3])
        metric = PrecisionAtK(k=30)  # top 30% = 3 samples
        result = metric(y_true, y_pred)
        assert result == 0.0

    def test_k_parameter_changes_result(self) -> None:
        y_true = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        y_pred = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])
        # top 10% = 1 sample → precision=1.0
        assert PrecisionAtK(k=10)(y_true, y_pred) == 1.0
        # top 50% = 5 samples → precision=2/5=0.4
        assert PrecisionAtK(k=50)(y_true, y_pred) == pytest.approx(0.4)

    def test_default_k(self) -> None:
        metric = PrecisionAtK()
        assert metric.k == 10

    def test_properties(self) -> None:
        metric = PrecisionAtK()
        assert metric.name == "precision_at_k"
        assert metric.needs_proba is True
        assert metric.greater_is_better is True

    def test_k_range_validation(self) -> None:
        with pytest.raises(ValueError, match="k must be in"):
            PrecisionAtK(k=0)
        with pytest.raises(ValueError, match="k must be in"):
            PrecisionAtK(k=101)

    def test_registered_for_binary(self) -> None:
        metrics = get_metrics_for_task(["precision_at_k"], "binary")
        assert len(metrics) == 1
        assert metrics[0].name == "precision_at_k"

    def test_multiclass_excluded(self) -> None:
        with pytest.raises(LizyMLError) as exc_info:
            get_metrics_for_task(["precision_at_k"], "multiclass")
        assert exc_info.value.code.value == "UNSUPPORTED_METRIC"

    def test_regression_excluded(self) -> None:
        with pytest.raises(LizyMLError) as exc_info:
            get_metrics_for_task(["precision_at_k"], "regression")
        assert exc_info.value.code.value == "UNSUPPORTED_METRIC"
