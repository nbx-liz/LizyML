"""float16 / longdouble arrivals on integer categories are compared as float64
(H-0127, #309).

A value float64 changes (a longdouble between two float64 values) equals no
category: it is unseen and follows ``unseen_policy``, reported with its own
text. A missing value is never compared: it stays missing under every policy.
"""

from __future__ import annotations

import typing
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import pytest

from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.features.encoders.categorical_encoder import (
    CategoricalEncoder,
    UnseenPolicy,
    comparable_values,
)

POLICIES: tuple[str, ...] = typing.get_args(UnseenPolicy)
KNOWN = [0, 1, 2, 3, 4]
MODE = 3

#: longdouble is wider than float64 here (float128 on x86-64 Linux, the CI).
WIDE_LONGDOUBLE = np.finfo(np.longdouble).eps < np.finfo(np.float64).eps
needs_wide = pytest.mark.skipif(
    not WIDE_LONGDOUBLE, reason="longdouble is float64 on this platform"
)

#: A longdouble next to the category 3 that float64 rounds onto 3.
NEAR_THREE = np.longdouble(3) + np.finfo(np.longdouble).eps * 3


def _encoder(policy: str) -> CategoricalEncoder:
    fit = pd.DataFrame({"a": pd.Series([0, 1, 2, 3, 3, 3, 4], dtype=object)})
    return CategoricalEncoder(unseen_policy=policy).fit(fit, ["a"])  # type: ignore[arg-type]


def _as_dtype(dtype: Any) -> Callable[[list[Any]], pd.Series]:
    return lambda values: pd.Series(values, dtype=dtype)


def _as_scalars(scalar: type) -> Callable[[list[Any]], pd.Series]:
    return lambda values: pd.Series([scalar(v) for v in values], dtype=object)


ARRIVALS: dict[str, Callable[[list[Any]], pd.Series]] = {
    "float16": _as_dtype(np.float16),
    "longdouble": _as_dtype(np.longdouble),
    "object of np.float16": _as_scalars(np.float16),
    "object of np.longdouble": _as_scalars(np.longdouble),
}


def _codes(
    encoder: CategoricalEncoder, series: pd.Series
) -> tuple[list[int], list[str]]:
    out, warnings = encoder.transform_with_warnings(pd.DataFrame({"a": series}))
    return out["a"].cat.codes.tolist(), warnings


@pytest.mark.parametrize("arrival", sorted(ARRIVALS))
@pytest.mark.parametrize(
    "values", [[0, 1, 2, 3, 4], [0, 1, 2.5, 4]], ids=["seen", "unseen"]
)
def test_an_arrival_encodes_as_its_float64_values(
    arrival: str, values: list[Any]
) -> None:
    """Criterion 1: the same values as float64 and as this arrival encode alike."""
    encoder = _encoder("mode")
    expected = _codes(encoder, pd.Series(values, dtype=np.float64))
    assert _codes(encoder, ARRIVALS[arrival](values)) == expected


def test_the_pandas_failure_is_what_the_normalisation_removes() -> None:
    """Should-fail witness: without the float64 step pandas refuses both types."""
    with pytest.raises(NotImplementedError):
        pd.Series([0, 1], dtype=np.float16).astype("category")
    if WIDE_LONGDOUBLE:
        with pytest.raises(ValueError):
            pd.Series([0, 1], dtype=np.longdouble).astype("category")


@needs_wide
@pytest.mark.parametrize("arrival", ["longdouble", "object of np.longdouble"])
@pytest.mark.parametrize("policy", POLICIES)
def test_a_value_float64_changes_is_unseen(arrival: str, policy: str) -> None:
    """Criterion 2: 3 + 3 eps never takes category 3's code."""
    assert np.float64(NEAR_THREE) == 3.0  # float64 alone would match category 3
    encoder = _encoder(policy)
    series = ARRIVALS[arrival]([0, 1, 2])
    series.iloc[2] = NEAR_THREE
    if policy == "error":
        with pytest.raises(LizyMLError) as exc:
            _codes(encoder, series)
        assert exc.value.code == ErrorCode.DATA_SCHEMA_INVALID
        assert exc.value.context["unseen_categories"] == [str(NEAR_THREE)]
        return
    codes, warnings = _codes(encoder, series)
    assert codes[:2] == [0, 1]
    assert codes[2] == (KNOWN.index(MODE) if policy == "mode" else -1)
    assert len(warnings) == 1 and str(NEAR_THREE) in warnings[0]


@pytest.mark.parametrize("arrival", sorted(ARRIVALS))
@pytest.mark.parametrize("policy", POLICIES)
def test_a_missing_value_stays_missing(arrival: str, policy: str) -> None:
    """Criterion 2b: NaN is not compared, so it is neither unseen nor reported."""
    encoder = _encoder(policy)
    series = ARRIVALS[arrival]([0, np.nan, 2])
    assert series.isna().tolist() == [False, True, False]
    codes, warnings = _codes(encoder, series)
    assert codes == [0, -1, 2]
    assert warnings == []


def test_other_columns_are_returned_unchanged() -> None:
    for series in (
        pd.Series([1.0, 2.0], dtype=np.float32),
        pd.Series(["a", "b"], dtype=object),
        pd.Series([1, "a"], dtype=object),
        pd.Series(pd.Categorical(["a", "b"])),
    ):
        values, changed = comparable_values(series)
        assert values is series
        assert not changed.any()


@pytest.mark.parametrize("policy", ["mode", "nan"])
def test_a_duplicate_index_keeps_rows_in_place(policy: str) -> None:
    """The replacement aligns masks by index; a repeated label must not move
    or duplicate rows."""
    encoder = _encoder(policy)
    series = pd.Series([0, 2.5, 1], dtype=np.float16, index=[0, 0, 1])
    out, warnings = encoder.transform_with_warnings(pd.DataFrame({"a": series}))
    assert out.index.tolist() == [0, 0, 1]
    unseen = KNOWN.index(MODE) if policy == "mode" else -1
    assert out["a"].cat.codes.tolist() == [0, unseen, 1]
    assert len(warnings) == 1 and "1 row(s)" in warnings[0]


@pytest.mark.parametrize("policy", ["mode", "nan"])
def test_a_tuple_category_is_replaced_whole(policy: str) -> None:
    """Rows are now replaced by mask (H-0127). The value replacement before it
    raised pandas' "Replacement lists must match in length" for a tuple mode."""
    cats = [(1, 2), (3, 4)]
    fit = pd.DataFrame({"a": pd.Categorical([(1, 2), (1, 2), (3, 4)], categories=cats)})
    encoder = CategoricalEncoder(unseen_policy=policy).fit(fit, ["a"])  # type: ignore[arg-type]
    arrived = pd.DataFrame({"a": pd.Series([(1, 2), (9, 9), (3, 4)], dtype=object)})
    out, warnings = encoder.transform_with_warnings(arrived)
    assert out["a"].cat.codes.tolist() == [0, 0 if policy == "mode" else -1, 1]
    assert len(warnings) == 1 and "(9, 9)" in warnings[0]
