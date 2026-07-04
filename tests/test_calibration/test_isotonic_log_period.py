"""Regression test: IsotonicCalibrator silences LightGBM via log_evaluation(period=-1).

The correct sentinel to disable per-iteration logging is ``period=-1``
(``period=0`` still logs). This asserts the implementation detail directly —
the effect (no eval log output) is not otherwise observable.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np

from lizyml.calibration.isotonic import IsotonicCalibrator


class TestIsotonicLogEvaluationPeriod:
    def test_period_is_negative_one(self) -> None:
        cal = IsotonicCalibrator()
        X = np.array([[0.1], [0.2], [0.3]])
        y = np.array([0.0, 1.0, 0.0])
        params: dict[str, object] = {"objective": "binary", "verbose": -1}
        with patch("lizyml.calibration.isotonic.lgbm.log_evaluation") as mock_log:
            mock_log.return_value = lambda *a, **kw: None
            cal._prepare_training(X, y, len(y), params)
            mock_log.assert_called_once_with(period=-1)
