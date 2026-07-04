"""Model.fit/evaluate smoke regressions from the code-review fixes.

- fit() returns a fully populated FitResult (calibrator + raw/calibrated metrics).
- RunMeta.timestamp carries UTC tz info.
- evaluate() on an unfitted model raises LizyMLError(MODEL_NOT_FIT), not AssertionError.
"""

from __future__ import annotations

import pytest

from lizyml import Model
from lizyml.core.exceptions import ErrorCode, LizyMLError
from tests._helpers import make_binary_df, make_config, make_regression_df


class TestFitResultImmutability:
    def test_fit_returns_complete_result(self) -> None:
        df = make_binary_df()
        m = Model(make_config("binary", calibration="platt", n_estimators=20))
        result = m.fit(data=df)
        assert result.calibrator is not None
        assert "raw" in result.metrics
        assert "calibrated" in result.metrics


class TestDatetimeUTC:
    def test_timestamp_has_utc(self) -> None:
        m = Model(make_config("regression", n_estimators=10))
        result = m.fit(data=make_regression_df(n=50))
        ts = result.run_meta.timestamp
        assert "+00:00" in ts or "Z" in ts, f"Timestamp lacks UTC: {ts}"


class TestEvaluateAssertReplaced:
    def test_evaluate_unfitted_raises_lizyml_error(self) -> None:
        m = Model(make_config("regression"))
        with pytest.raises(LizyMLError) as exc_info:
            m.evaluate()
        assert exc_info.value.code == ErrorCode.MODEL_NOT_FIT
