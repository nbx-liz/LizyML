"""Tests for Confusion Matrix table (H-0016)."""

from __future__ import annotations

import pandas as pd
import pytest

from lizyml import Model
from lizyml.core.exceptions import ErrorCode, LizyMLError
from tests._helpers import (
    make_binary_df,
    make_config,
    make_multiclass_df,
    make_regression_df,
)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestConfusionMatrixBinary:
    def test_returns_dict_with_is_oos(self) -> None:
        m = Model(make_config("binary"))
        m.fit(data=make_binary_df())
        result = m.confusion_matrix()
        assert "is" in result
        assert "oos" in result

    def test_oos_shape_binary(self) -> None:
        m = Model(make_config("binary"))
        m.fit(data=make_binary_df())
        result = m.confusion_matrix()
        assert result["oos"].shape == (2, 2)

    def test_is_shape_binary(self) -> None:
        m = Model(make_config("binary"))
        m.fit(data=make_binary_df())
        result = m.confusion_matrix()
        assert result["is"].shape == (2, 2)

    def test_is_dataframe(self) -> None:
        m = Model(make_config("binary"))
        m.fit(data=make_binary_df())
        result = m.confusion_matrix()
        assert isinstance(result["oos"], pd.DataFrame)
        assert isinstance(result["is"], pd.DataFrame)

    def test_threshold_changes_result(self) -> None:
        m = Model(make_config("binary"))
        m.fit(data=make_binary_df())
        result_low = m.confusion_matrix(threshold=0.1)
        result_high = m.confusion_matrix(threshold=0.9)
        # Different thresholds should produce different confusion matrices
        assert not result_low["oos"].equals(result_high["oos"])


class TestConfusionMatrixMulticlass:
    def test_shape_multiclass(self) -> None:
        m = Model(make_config("multiclass"))
        m.fit(data=make_multiclass_df(n=200))
        result = m.confusion_matrix()
        assert result["oos"].shape == (3, 3)
        assert result["is"].shape == (3, 3)


class TestConfusionMatrixErrors:
    def test_regression_raises(self) -> None:
        m = Model(make_config("regression"))
        m.fit(data=make_regression_df(n=100))
        with pytest.raises(LizyMLError) as exc_info:
            m.confusion_matrix()
        assert exc_info.value.code == ErrorCode.UNSUPPORTED_TASK

    def test_before_fit_raises(self) -> None:
        m = Model(make_config("binary"))
        with pytest.raises(LizyMLError) as exc_info:
            m.confusion_matrix()
        assert exc_info.value.code == ErrorCode.MODEL_NOT_FIT
