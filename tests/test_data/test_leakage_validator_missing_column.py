"""Leakage checks refuse a frame without their named column (H-0112, #311).

``validate_no_target_leakage`` and ``validate_time_series_order`` used to
return ``[]`` when their column was missing -- the same answer as a frame that
was fully checked and found clean. A misspelt target or time column therefore
got a "clean" result from a check that compared nothing.
"""

from __future__ import annotations

import pandas as pd
import pytest

from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.data import validate_no_target_leakage, validate_time_series_order

_LEAKING = pd.DataFrame({"a": [1, 2, 3], "y": [1, 2, 3]})  # "a" duplicates "y"
_UNSORTED = pd.DataFrame({"t": [3, 1, 2], "x": [0.1, 0.2, 0.3]})


@pytest.mark.parametrize("raise_on_violation", [True, False])
@pytest.mark.parametrize("target", ["Y", "nonexistent"])
def test_missing_target_raises(target: str, raise_on_violation: bool) -> None:
    with pytest.raises(LizyMLError) as exc:
        validate_no_target_leakage(
            _LEAKING, target, raise_on_violation=raise_on_violation
        )
    assert exc.value.code == ErrorCode.DATA_SCHEMA_INVALID
    assert exc.value.context["target"] == target
    assert exc.value.context["missing_columns"] == [target]
    assert exc.value.context["available_columns"] == ["a", "y"]
    assert target in exc.value.user_message


@pytest.mark.parametrize("raise_on_violation", [True, False])
@pytest.mark.parametrize("time_col", ["T", "nonexistent"])
def test_missing_time_col_raises(time_col: str, raise_on_violation: bool) -> None:
    with pytest.raises(LizyMLError) as exc:
        validate_time_series_order(
            _UNSORTED, time_col, raise_on_violation=raise_on_violation
        )
    assert exc.value.code == ErrorCode.DATA_SCHEMA_INVALID
    assert exc.value.context["time_col"] == time_col
    assert exc.value.context["missing_columns"] == [time_col]
    assert exc.value.context["available_columns"] == ["t", "x"]
    assert time_col in exc.value.user_message


def test_present_target_behaviour_unchanged() -> None:
    with pytest.raises(LizyMLError) as exc:
        validate_no_target_leakage(_LEAKING, "y")
    assert exc.value.code == ErrorCode.LEAKAGE_SUSPECTED
    assert len(validate_no_target_leakage(_LEAKING, "y", raise_on_violation=False)) == 1
    assert (
        validate_no_target_leakage(pd.DataFrame({"a": [3, 1, 2], "y": [1, 2, 3]}), "y")
        == []
    )


def test_present_time_col_behaviour_unchanged() -> None:
    with pytest.raises(LizyMLError) as exc:
        validate_time_series_order(_UNSORTED, "t")
    assert exc.value.code == ErrorCode.LEAKAGE_SUSPECTED
    assert (
        len(validate_time_series_order(_UNSORTED, "t", raise_on_violation=False)) == 1
    )
    assert validate_time_series_order(_UNSORTED.sort_values("t"), "t") == []
