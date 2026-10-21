"""H-0139 acceptance 1: the temporal identity key (#379).

Closed fixtures of pairs the decided contract joins and pairs it keeps
apart, for the library key and its copy in the generated scripts, and the
key's stability across the codec, the generated codec and joblib.
"""

from __future__ import annotations

import datetime as dt
import io
import zoneinfo
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pytest

from lizyml.core.category_temporal import CalendarDuration, calendar_units
from lizyml.core.category_values import category_identity_key, decode, encode, refusal
from tests.test_codegen.test_h0133_codec_copy import COPY
from tests.test_core.test_category_values import SAMPLES

NY = zoneinfo.ZoneInfo("America/New_York")
UTC = dt.timezone.utc
EST = dt.timezone(dt.timedelta(hours=-5))
FOLD0 = dt.datetime(2024, 11, 3, 1, 30, fold=0, tzinfo=NY)
FOLD1 = FOLD0.replace(fold=1)
T0 = pd.Timestamp("2024-01-02 03:04:05")
I64 = np.iinfo(np.int64).max

UTC8 = pd.Timestamp("2024-01-02 08:00", tz="UTC")
NY3 = pd.Timestamp("2024-01-02 03:00", tz=NY)

EQUAL = [
    # aware: one UTC instant, whatever the time zone's provider
    (UTC8, NY3),
    (pd.Timestamp(dt.datetime(2024, 1, 2, 3, tzinfo=EST)), NY3),
    (dt.datetime(2024, 1, 2, 8, tzinfo=UTC), UTC8),
    (pd.Timestamp(FOLD1), dt.datetime(2024, 11, 3, 1, 30, tzinfo=EST)),
    (FOLD1, pd.Timestamp(FOLD1)),
    # naive: one wall-clock coordinate, whatever the type or unit
    (T0.as_unit("s"), T0.as_unit("ns")),
    (T0, T0.to_pydatetime()),
    (np.datetime64("2024-01-02", "D"), np.datetime64("2024-01-02T00:00:00", "ns")),
    (np.datetime64("2024-01-02T03:04:05", "ns"), T0),
    (np.datetime64(54, "Y"), np.datetime64("2024-01-01", "D")),
    (np.datetime64(648, "M"), np.datetime64("2024-01-01", "D")),
    (np.datetime64(-1, "M"), np.datetime64("1969-12-01", "D")),
    (np.datetime64(1, "W"), np.datetime64(7, "D")),
    (np.datetime64(4, "2s"), np.datetime64(8, "s")),
    (np.datetime64(1000, "fs"), np.datetime64(1, "ps")),
    (np.datetime64(-1, "D"), np.datetime64("1969-12-31", "D")),
    # fixed durations: one length
    (np.timedelta64(1, "h"), np.timedelta64(60, "m")),
    (pd.Timedelta(hours=1), dt.timedelta(hours=1)),
    (pd.Timedelta(hours=1).as_unit("s"), np.timedelta64(3600, "s")),
    (np.timedelta64(3, "15m"), np.timedelta64(45, "m")),
    (np.timedelta64(1, "W"), dt.timedelta(days=7)),
    # containers recurse
    ((UTC8, 1), (NY3, 1)),
    (frozenset({np.timedelta64(1, "h")}), frozenset({dt.timedelta(hours=1)})),
]  # fmt: skip

DISTINCT = [
    (pd.Timestamp(FOLD0), pd.Timestamp(FOLD1)),  # two instants
    (FOLD0, FOLD1),  # Python's == calls them equal; they are two instants
    (T0, T0.tz_localize("UTC")),  # naive and aware
    (T0.to_pydatetime(), T0.tz_localize("UTC").to_pydatetime()),
    (dt.date(2024, 1, 2), pd.Timestamp("2024-01-02")),  # date and midnight
    (dt.date(2024, 1, 2), dt.datetime(2024, 1, 2)),
    (np.datetime64(4, "ps"), np.datetime64(4, "fs")),  # sub-ns kept exact
    (np.datetime64(4, "as"), np.datetime64(0, "ns")),
    (np.timedelta64(0, "s"), np.datetime64(0, "s")),  # duration and datetime
    (dt.timedelta(0), 0),  # duration and number
    (pd.Timestamp("1970-01-01"), 0),
    (np.timedelta64(1, "s"), 1),
    (np.datetime64(4, "D"), np.datetime64(5, "D")),
]


def _pytz_pairs() -> tuple[list[tuple[Any, Any]], list[tuple[Any, Any]]]:
    try:
        import pytz  # type: ignore[import-untyped, unused-ignore]
    except ImportError:
        return [], []
    ny = pytz.timezone("America/New_York")
    std = pd.Timestamp(ny.localize(dt.datetime(2024, 11, 3, 1, 30), is_dst=False))
    dst = pd.Timestamp(ny.localize(dt.datetime(2024, 11, 3, 1, 30), is_dst=True))
    plain = pd.Timestamp(ny.localize(dt.datetime(2024, 1, 2, 3, 0)))
    equal = [
        (std, pd.Timestamp(FOLD1)),
        (plain, pd.Timestamp(dt.datetime(2024, 1, 2, 3, tzinfo=NY))),
    ]
    return equal, [(dst, std)]  # fmt: skip


PYTZ_EQUAL, PYTZ_DISTINCT = _pytz_pairs()


@pytest.mark.parametrize(("a", "b"), EQUAL + PYTZ_EQUAL, ids=repr)
def test_the_contract_joins_these_pairs(a: Any, b: Any) -> None:
    assert category_identity_key(a) == category_identity_key(b)
    assert COPY._identity_key(a) == COPY._identity_key(b)


@pytest.mark.parametrize(("a", "b"), DISTINCT + PYTZ_DISTINCT, ids=repr)
def test_the_contract_keeps_these_pairs_apart(a: Any, b: Any) -> None:
    assert category_identity_key(a) != category_identity_key(b)
    assert COPY._identity_key(a) != COPY._identity_key(b)


def test_the_extremes_have_exact_coordinates() -> None:
    low = dt.datetime.min.replace(tzinfo=dt.timezone(dt.timedelta(hours=23)))
    high = dt.datetime.max.replace(tzinfo=dt.timezone(dt.timedelta(hours=-23)))
    assert category_identity_key(low) == (
        "datetime",
        "aware",
        -62135679600000000000000000000,
    )
    assert category_identity_key(high) == (
        "datetime",
        "aware",
        253402383599999999000000000000,
    )
    assert category_identity_key(np.datetime64(I64, "ns")) == (
        "datetime",
        "naive",
        I64 * 10**9,
    )
    assert category_identity_key(np.timedelta64(-I64, "W")) == (
        "duration",
        -I64 * 604800 * 10**18,
    )


_FIXTURE_VALUES = [
    v for pair in EQUAL + DISTINCT + PYTZ_EQUAL + PYTZ_DISTINCT for v in pair
]
_TEMPORAL_SAMPLES = [v for vs in SAMPLES.values() for v in vs if not calendar_units(v)]


@pytest.mark.parametrize("value", _FIXTURE_VALUES + _TEMPORAL_SAMPLES, ids=repr)
def test_the_key_survives_every_serializer(value: Any) -> None:
    key = category_identity_key(value)
    buffer = io.BytesIO()
    joblib.dump(value, buffer, compress=3)
    buffer.seek(0)
    assert category_identity_key(joblib.load(buffer)) == key
    assert category_identity_key(decode(encode(value))) == key
    assert COPY._identity_key(COPY._decode_value(COPY._encode_value(value))) == key
    assert COPY._identity_key(value) == key


@pytest.mark.parametrize("unit", ["Y", "M"])
@pytest.mark.parametrize("wrap", ["scalar", "tuple", "frozenset", "multiple"])
def test_a_calendar_duration_has_no_key_anywhere(unit: str, wrap: str) -> None:
    value: Any = np.timedelta64(2, unit)
    if wrap == "tuple":
        value = ("a", (value,))
    elif wrap == "frozenset":
        value = frozenset({value, 1})
    elif wrap == "multiple":
        value = np.timedelta64(3, f"2{unit}")
    assert refusal(value) is None  # still in V: saved models keep loading
    assert calendar_units(value) == {unit}
    with pytest.raises(CalendarDuration):
        category_identity_key(value)
    with pytest.raises(TypeError):
        COPY._identity_key(value)


def _generic_datetime64() -> Any:
    """A unitless datetime64 (its repr raises; NumPy builds none from an int)."""
    return np.array(4, dtype=np.int64).view("datetime64")[()]


@pytest.mark.parametrize(
    "build",
    [
        _generic_datetime64,
        lambda: np.timedelta64(4),
        lambda: (_generic_datetime64(), 1),
        # NumPy 2 cannot hash a unitless timedelta64, so only the datetime64
        # goes into the frozenset.
        lambda: frozenset({_generic_datetime64(), 1}),
        lambda: (np.timedelta64(4), "a"),
    ],
    ids=[
        "datetime64",
        "timedelta64",
        "datetime64 in a tuple",
        "datetime64 in a frozenset",
        "timedelta64 in a tuple",
    ],
)
def test_a_unitless_value_survives_every_serializer(build: Any) -> None:
    value = build()
    assert refusal(value) is None
    key = category_identity_key(value)
    buffer = io.BytesIO()
    joblib.dump(value, buffer, compress=3)
    buffer.seek(0)
    assert category_identity_key(joblib.load(buffer)) == key
    assert category_identity_key(decode(encode(value))) == key
    assert COPY._identity_key(COPY._decode_value(COPY._encode_value(value))) == key
    assert COPY._identity_key(value) == key


def test_a_unitless_datetime64_is_apart_from_every_unit() -> None:
    key = category_identity_key(_generic_datetime64())
    assert key == ("datetime", "naive", ("datetime64-generic", 4))
    assert key != category_identity_key(np.datetime64(4, "s"))


def test_a_unitless_timedelta64_is_apart_from_every_fixed_duration() -> None:
    generic = np.timedelta64(4)
    if refusal(generic) is not None:
        pytest.skip(f"not in V: {refusal(generic)}")
    key = category_identity_key(generic)
    assert key == ("timedelta64-generic", 4)
    assert key != category_identity_key(np.timedelta64(4, "s"))
    assert COPY._identity_key(generic) == key
