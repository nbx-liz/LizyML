"""Leakage checks refuse a frame without their named column (H-0112, #311).

``validate_no_target_leakage`` and ``validate_time_series_order`` used to
return ``[]`` when their column was missing -- the same answer as a frame that
was fully checked and found clean. A misspelt target or time column therefore
got a "clean" result from a check that compared nothing.
"""

from __future__ import annotations

from typing import Any

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


_MULTI = pd.DataFrame(
    [[1, 1], [2, 2], [3, 3]],
    columns=pd.MultiIndex.from_tuples([("a", "x"), ("b", "y")]),
)


@pytest.mark.parametrize("raise_on_violation", [True, False])
@pytest.mark.parametrize(
    "validator, role",
    [
        (validate_no_target_leakage, "target"),
        (validate_time_series_order, "time_col"),
    ],
    ids=["leakage", "time_order"],
)
def test_partial_multiindex_key_is_missing(
    validator: Any, role: str, raise_on_violation: bool
) -> None:
    """``"a" in df.columns`` is true for a partial MultiIndex key (review r1)."""
    with pytest.raises(LizyMLError) as exc:
        validator(_MULTI, "a", raise_on_violation=raise_on_violation)
    assert exc.value.code == ErrorCode.DATA_SCHEMA_INVALID
    assert exc.value.context[role] == "a"
    assert exc.value.context["missing_columns"] == ["a"]
    assert exc.value.context["available_columns"] == [("a", "x"), ("b", "y")]


def test_complete_multiindex_label_is_present() -> None:
    with pytest.raises(LizyMLError) as exc:
        validate_no_target_leakage(_MULTI, ("a", "x"))
    assert exc.value.code == ErrorCode.LEAKAGE_SUSPECTED
    assert validate_time_series_order(_MULTI, ("a", "x")) == []


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
