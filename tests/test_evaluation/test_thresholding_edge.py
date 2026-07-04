"""Edge-case coverage for evaluation/thresholding.py."""

from __future__ import annotations

from typing import Any

import numpy as np

from lizyml.evaluation.thresholding import optimise_threshold


class TestOptimiseThreshold:
    def test_minimize(self) -> None:
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.1, 0.4, 0.6, 0.9])

        def zero_one_loss(y_t: Any, y_p: Any) -> float:
            return float(np.mean(y_t != y_p))

        best_thresh, best_score = optimise_threshold(
            y_true, y_proba, zero_one_loss, greater_is_better=False
        )
        assert best_score <= 0.5
        assert 0.0 <= best_thresh <= 1.0
