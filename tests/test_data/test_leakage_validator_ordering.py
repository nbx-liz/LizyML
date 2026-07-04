"""Regression tests for the target-leakage NaN-ordering guard (BUG-3, 2026-04-10).

The NaN-position guard (``isna().equals``) must be evaluated before
``np.allclose``; otherwise columns with a differing NaN count reach ``dropna()``
with mismatched lengths and raise ``ValueError`` (historically swallowed).

These assert observable behavior — the ``_series_perfectly_correlated`` helper's
boolean result and the public validator's flagging — instead of patching
``np.allclose``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.data.validators import (
    _series_perfectly_correlated,
    validate_no_target_leakage,
)


class TestSeriesPerfectlyCorrelated:
    """The pure helper encodes the guard ordering."""

    def test_exact_copy_is_correlated(self) -> None:
        y = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0])
        col = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0])
        assert _series_perfectly_correlated(col, y) is True

    def test_different_nan_positions_not_correlated(self) -> None:
        y = pd.Series([1.0, 2.0, 3.0, np.nan, 5.0])
        col = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0])
        assert _series_perfectly_correlated(col, y) is False

    def test_mismatched_nan_count_short_circuits_without_error(self) -> None:
        """Differing NaN COUNT would make ``np.allclose(dropna, dropna)`` raise
        ValueError if the guard order regressed. The helper never catches
        exceptions, so a wrong order surfaces here as a test error, not a
        silent pass."""
        y = pd.Series([1.0, 2.0, 3.0, np.nan, 5.0])  # 1 NaN -> dropna len 4
        col = pd.Series([1.0, np.nan, np.nan, 4.0, 5.0])  # 2 NaNs -> dropna len 3
        assert _series_perfectly_correlated(col, y) is False

    def test_non_numeric_not_correlated(self) -> None:
        y = pd.Series([1.0, 2.0, 3.0])
        col = pd.Series(["a", "b", "c"])
        assert _series_perfectly_correlated(col, y) is False


class TestLeakageValidatorNaNOrder:
    """Public validator: differing NaN layouts must not be flagged or error."""

    def test_different_nan_positions_not_flagged(self) -> None:
        df = pd.DataFrame(
            {
                "target": [1.0, 2.0, 3.0, np.nan, 5.0],
                "tricky": [1.0, 2.0, np.nan, 4.0, 5.0],
            }
        )
        result = validate_no_target_leakage(df, "target", raise_on_violation=False)
        assert result == []

    def test_mismatched_nan_count_not_flagged(self) -> None:
        df = pd.DataFrame(
            {
                "target": [1.0, 2.0, 3.0, np.nan, 5.0],
                "tricky": [1.0, np.nan, np.nan, 4.0, 5.0],
            }
        )
        result = validate_no_target_leakage(df, "target", raise_on_violation=False)
        assert result == []

    def test_real_leakage_with_matching_nan_still_detected(self) -> None:
        df = pd.DataFrame(
            {
                "target": [1.0, 2.0, np.nan, 4.0, 5.0],
                "leak": [1.0, 2.0, np.nan, 4.0, 5.0],
            }
        )
        with pytest.raises(LizyMLError) as exc_info:
            validate_no_target_leakage(df, "target", raise_on_violation=True)
        assert exc_info.value.code == ErrorCode.LEAKAGE_SUSPECTED
