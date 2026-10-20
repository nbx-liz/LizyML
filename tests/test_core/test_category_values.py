"""H-0133 P1-P2: the closed set V of saveable category values and its codec.

The admitted population is derived from ``ADMITTED_TYPES`` (criterion 1): a
sample factory covers every listed type, and a test fails when a type is
added to the list without samples. Every sample round-trips through JSON with
an equal value, an equal hash and the same type (criterion 4), except the
three documented Python-type returns.
"""

from __future__ import annotations

import datetime as dt
import decimal
import json
import subprocess
import sys
import zoneinfo
from typing import Any

import numpy as np
import pandas as pd
import pytest

from lizyml.core.category_values import (
    ADMITTED_TYPES,
    MAX_DEPTH,
    decode,
    encode,
    refusal,
)
from lizyml.core.exceptions import ErrorCode, LizyMLError

NY = zoneinfo.ZoneInfo("America/New_York")
TOKYO = zoneinfo.ZoneInfo("Asia/Tokyo")
FIXED = dt.timezone(dt.timedelta(hours=-5, microseconds=1))
EPOCH = dt.datetime(1970, 1, 1)


def _nested(depth: int) -> Any:
    value: Any = 1
    for _ in range(depth):
        value = (value,)
    return value


_DT64_UNITS = ("Y", "M", "W", "D", "h", "m", "s", "ms", "us", "ns", "ps", "fs", "as")

#: Samples per admitted type: the axes criterion 1 names.
SAMPLES: dict[type, list[Any]] = {
    str: ["a", "", "日本"],
    int: [0, -3, 2**70],
    float: [1.5, -0.0, float("inf"), float("-inf")],
    bool: [True, False],
    bytes: [b"a", b"", bytes(range(256))],
    decimal.Decimal: [
        decimal.Decimal("1.50"),
        decimal.Decimal("-0"),
        decimal.Decimal("Infinity"),
    ],
    dt.date: [dt.date(2024, 1, 1), dt.date.min, dt.date.max],
    dt.datetime: [
        dt.datetime(2024, 1, 1, 12, 30, 15, 123456),
        dt.datetime(2024, 11, 3, 1, 30, fold=0, tzinfo=NY),
        dt.datetime(2024, 11, 3, 1, 30, fold=1, tzinfo=NY),
        dt.datetime(2024, 1, 1, tzinfo=FIXED),
        dt.datetime.min.replace(tzinfo=dt.timezone(dt.timedelta(hours=23))),
        dt.datetime.max.replace(tzinfo=dt.timezone(dt.timedelta(hours=-23))),
    ],
    dt.timedelta: [dt.timedelta(days=-2, seconds=5, microseconds=7), dt.timedelta(0)],
    complex: [complex(1, -2), complex(float("inf"), 0)],
    pd.Timestamp: [
        *(
            pd.Timestamp("2024-01-01 01:02:03.456789").as_unit(u)
            for u in ("s", "ms", "us", "ns")
        ),
        pd.Timestamp(dt.datetime(2024, 11, 3, 1, 30, fold=0, tzinfo=NY)).as_unit("ns"),
        pd.Timestamp(dt.datetime(2024, 11, 3, 1, 30, fold=1, tzinfo=NY)),
        pd.Timestamp("2024-01-01", tz=zoneinfo.ZoneInfo.no_cache("Asia/Tokyo")),
        pd.Timestamp("2024-01-01", tz=FIXED),
        pd.Timestamp("2024-01-01", tz=dt.timezone.utc),
    ],
    pd.Timedelta: [
        *(pd.Timedelta(5, unit="s").as_unit(u) for u in ("s", "ms", "us", "ns")),
        pd.Timedelta(7, unit="D"),
    ],
    np.bool_: [np.bool_(True), np.bool_(False)],
    np.int8: [np.int8(-3), np.int8(5)],
    np.int16: [np.int16(300), np.int16(-2)],
    np.int32: [np.int32(-70000), np.int32(9)],
    np.int64: [np.int64(2**40), np.int64(-1)],
    np.uint8: [np.uint8(200), np.uint8(1)],
    np.uint16: [np.uint16(60000), np.uint16(2)],
    np.uint32: [np.uint32(4000000000), np.uint32(3)],
    np.uint64: [np.uint64(2**63 + 5), np.uint64(4)],
    np.float32: [np.float32(0.1), np.float32(-2.5)],
    np.float64: [np.float64(-0.0), np.float64(1.25)],
    np.complex64: [np.complex64(1 + 2j), np.complex64(-1)],
    np.complex128: [np.complex128(0.1 - 3j), np.complex128(2)],
    np.str_: [np.str_("x"), np.str_("yz")],
    np.bytes_: [np.bytes_(b"y"), np.bytes_(b"zz")],
    np.datetime64: [
        *(np.datetime64(4, u) for u in _DT64_UNITS),
        np.datetime64(4, "2s"),
        np.datetime64(3, "10ms"),
    ],
    np.timedelta64: [
        *(np.timedelta64(4, u) for u in _DT64_UNITS),
        np.timedelta64(3, "15m"),
    ],
    tuple: [
        (1, "a", b"b"),
        (),
        (pd.Timestamp("2024-01-01"), (decimal.Decimal("2"),)),
        _nested(MAX_DEPTH),
    ],
    frozenset: [
        frozenset({1, "a", b"c", 2.5}),
        frozenset(),
        frozenset({(1, 2), frozenset({3})}),
    ],
}

try:  # pandas 2.x (the dependency floor) makes pytz zones from string timezones
    import pytz
except ImportError:  # pandas 3 does not depend on pytz
    pytz = None
if pytz is not None:
    _ny = pytz.timezone("America/New_York")
    SAMPLES[pd.Timestamp] += [
        pd.Timestamp(_ny.localize(dt.datetime(2024, 11, 3, 1, 30), is_dst=dst))
        for dst in (True, False)
    ]

#: Admitted values that come back as a Python type (P2), keyed by original type.
PYTHON_TYPE_RETURNS = {
    np.bool_: bool,
    np.int8: int, np.int16: int, np.int32: int, np.int64: int,
    np.uint8: int, np.uint16: int, np.uint32: int, np.uint64: int,
    np.float32: float, np.float64: float,
    np.str_: str, np.bytes_: bytes,
    np.complex64: complex, np.complex128: complex,
}  # fmt: skip


def _all_samples() -> list[tuple[str, Any]]:
    return [
        (f"{t.__name__}[{i}]", v) for t, vs in SAMPLES.items() for i, v in enumerate(vs)
    ]


def _round_trip(value: Any) -> Any:
    return decode(json.loads(json.dumps(encode(value))))


def test_the_sample_population_covers_every_admitted_type() -> None:
    assert set(SAMPLES) == set(ADMITTED_TYPES)
    assert all(SAMPLES[t] for t in SAMPLES)


@pytest.mark.parametrize(
    ("name", "value"), _all_samples(), ids=[n for n, _ in _all_samples()]
)
def test_every_admitted_value_is_admitted(name: str, value: Any) -> None:
    assert refusal(value) is None


@pytest.mark.parametrize(
    ("name", "value"), _all_samples(), ids=[n for n, _ in _all_samples()]
)
def test_every_admitted_value_round_trips_equal_with_an_equal_hash(
    name: str, value: Any
) -> None:
    back = _round_trip(value)
    assert back == value
    assert hash(back) == hash(value)
    assert type(back) is PYTHON_TYPE_RETURNS.get(type(value), type(value))


def test_a_timestamp_keeps_its_unit() -> None:
    for unit in ("s", "ms", "us", "ns"):
        assert _round_trip(pd.Timestamp("2024-01-01").as_unit(unit)).unit == unit


def test_a_datetime64_keeps_its_unit_and_multiplier() -> None:
    value = np.datetime64(4, "2s")
    back = _round_trip(value)
    assert np.datetime_data(back.dtype) == np.datetime_data(value.dtype)


def test_a_fold_datetime_keeps_its_fold_and_matches_a_fresh_value() -> None:
    value = dt.datetime(2024, 11, 3, 1, 30, fold=1, tzinfo=NY)
    back = _round_trip(value)
    assert back.fold == 1
    fresh = dt.datetime(
        2024, 11, 3, 1, 30, fold=1, tzinfo=zoneinfo.ZoneInfo("America/New_York")
    )
    assert back in {fresh}


def test_a_timestamp_and_an_equal_datetime_match_in_a_set() -> None:
    """Pandas merges them at fit; the restored representative still matches both."""
    back = _round_trip(pd.Timestamp("2024-01-01 09:00", tz=TOKYO))
    assert back in {dt.datetime(2024, 1, 1, 9, 0, tzinfo=TOKYO)}


def test_signed_zero_survives() -> None:
    assert str(_round_trip(-0.0)) == "-0.0"
    assert str(_round_trip(complex(-0.0, 1)).real) == "-0.0"


@pytest.mark.parametrize(
    ("value", "reason"),
    [
        ((1, float("nan")), "NaN"),
        ((1, np.float64("nan")), "NaN"),
        (decimal.Decimal("NaN"), "NaN"),
        (decimal.Decimal("sNaN"), "NaN"),
        (complex(float("nan"), 0), "NaN"),
        (np.complex128(complex(0, float("nan"))), "NaN"),
        (np.datetime64("NaT"), "NaT"),
        (np.timedelta64("NaT"), "NaT"),
        (pd.NaT, "type"),
        (_nested(MAX_DEPTH + 1), "nested deeper"),
        (type("S", (str,), {})("a"), "type"),
        (type("I", (int,), {})(1), "type"),
        (np.longdouble(1), "type"),
        (np.float16(1), "type"),
        (object(), "type"),
        ([1], "type"),
        (
            pd.Timestamp("2024-01-01", tz="UTC")
            .tz_convert(dt.timezone.utc)
            .replace(tzinfo=None),
            None,
        ),
        (
            dt.datetime(2024, 1, 1, tzinfo=zoneinfo.ZoneInfo.no_cache("Asia/Tokyo")),
            "tzinfo",
        ),
    ],
)
def test_values_outside_v_are_refused(value: Any, reason: str | None) -> None:
    got = refusal(value)
    if reason is None:
        assert got is None
    else:
        assert got is not None and reason in got


def test_a_caller_tzinfo_is_refused_without_running_its_code() -> None:
    calls: list[str] = []

    class CallerTz(dt.tzinfo):
        @property
        def zone(self) -> str:
            calls.append("zone")
            return "Asia/Tokyo"

        def utcoffset(self, d: dt.datetime | None) -> dt.timedelta:
            calls.append("utcoffset")
            return dt.timedelta(hours=9)

    assert (
        refusal(pd.Timestamp("2024-01-01", tz=dt.timezone.utc).tz_convert(TOKYO))
        is None
    )
    caller_dt = dt.datetime(2024, 1, 1, tzinfo=CallerTz())
    assert refusal(caller_dt) is not None
    assert calls == []


def _subclass_instances() -> dict[str, Any]:
    """An instance of a subclass of every admitted type Python lets one
    subclass, built from that type's first sample."""
    built: dict[str, Any] = {}
    for kind in ADMITTED_TYPES:
        if not kind.__flags__ & (1 << 10):  # Py_TPFLAGS_BASETYPE
            continue
        sub = type(f"Sub{kind.__name__}", (kind,), {})
        sample = SAMPLES[kind][0]
        args = _SUBCLASS_ARGS.get(kind, (sample,))
        instance = sub(*args)
        # Some constructors (pandas, numpy datetime scalars, np.bool_) return
        # the base type whatever class is called: no subclass value exists.
        if type(instance) is sub:
            built[kind.__name__] = instance
        else:
            BASE_ONLY.add(kind.__name__)
    return built


#: Subclassable admitted types whose constructor returns the base type.
BASE_ONLY: set[str] = set()


#: Constructor arguments where a type is not built from one of its values.
_SUBCLASS_ARGS: dict[type, tuple[Any, ...]] = {
    dt.date: (2024, 1, 2),
    dt.datetime: (2024, 1, 2, 3),
    dt.timedelta: (1, 2),
    tuple: ([1, 2],),
    frozenset: ([1, 2],),
}


SUBCLASS_INSTANCES = _subclass_instances()


def test_the_subclass_population_covers_the_subclassable_types() -> None:
    subclassable = {k.__name__ for k in ADMITTED_TYPES if k.__flags__ & (1 << 10)}
    assert set(SUBCLASS_INSTANCES) | BASE_ONLY == subclassable
    assert not set(SUBCLASS_INSTANCES) & BASE_ONLY
    # The Python types build real subclass values; the population is not empty.
    assert {"str", "int", "float", "bytes", "Decimal", "tuple", "frozenset"} <= set(
        SUBCLASS_INSTANCES
    )


@pytest.mark.parametrize("name", sorted(SUBCLASS_INSTANCES))
def test_a_subclass_of_every_admitted_type_is_refused(name: str) -> None:
    value = SUBCLASS_INSTANCES[name]
    assert refusal(value) is not None
    assert refusal((value,)) is not None


def test_a_caller_subclass_is_refused_without_running_its_code() -> None:
    class Loud(str):
        def __eq__(self, other: object) -> bool:
            raise AssertionError("caller __eq__ ran")

        __hash__ = str.__hash__

    assert refusal(Loud("a")) is not None
    assert refusal((Loud("a"),)) is not None


def test_encoding_a_refused_value_raises_serialization_failed() -> None:
    with pytest.raises(LizyMLError) as caught:
        encode((1, float("nan")))
    assert caught.value.code == ErrorCode.SERIALIZATION_FAILED
    assert "NaN" in caught.value.context["reason"]


@pytest.mark.parametrize(
    "obj",
    [
        {"$lizyml": "bytes"},
        {"$lizyml": "bytes", "b64": "YQ==", "extra": 1},
        {"$lizyml": "unknown", "x": 1},
        {"no_tag": 1},
        {"$lizyml": "decimal", "str": 3},
        {"$lizyml": "datetime", "us": 0, "fold": 2, "tz": None},
        {
            "$lizyml": "datetime",
            "us": 0,
            "fold": 0,
            "tz": {"kind": "iana", "name": "No/Such"},
        },
        {"$lizyml": "timestamp", "unit": "D", "value": 0, "tz": None},
        {"$lizyml": "timestamp", "unit": "s", "value": 0, "tz": {"kind": "offset"}},
        {"$lizyml": "datetime64", "unit": "s", "count": 0, "value": 1},
        {"$lizyml": "tuple", "items": "ab"},
        {"$lizyml": "complex", "real": True, "imag": 0.0},
        [1, 2],
        None,
        # Well-formed shapes that decode outside V (encode never writes them).
        float("nan"),
        {"$lizyml": "decimal", "str": "NaN"},
        {"$lizyml": "decimal", "str": "sNaN"},
        {"$lizyml": "complex", "real": float("nan"), "imag": 0.0},
        {"$lizyml": "datetime64", "unit": "s", "count": 1, "value": -(2**63)},
        {"$lizyml": "tuple", "items": [1, float("nan")]},
        # Decodable but not canonical (implementation review round 2).
        {"$lizyml": "timedelta", "days": 0, "seconds": 86400, "microseconds": 0},
        {"$lizyml": "frozenset", "items": [1, 1]},
        {"$lizyml": "frozenset", "items": [2, 1]},
    ],
)
def test_a_malformed_tag_is_refused(obj: Any) -> None:
    with pytest.raises(LizyMLError) as caught:
        decode(obj)
    assert caught.value.code == ErrorCode.DESERIALIZATION_FAILED


def test_decoding_does_not_nest_past_the_limit() -> None:
    obj: Any = 1
    for _ in range(MAX_DEPTH + 1):
        obj = {"$lizyml": "tuple", "items": [obj]}
    with pytest.raises(LizyMLError) as caught:
        decode(obj)
    assert caught.value.code == ErrorCode.DESERIALIZATION_FAILED


def test_a_frozenset_is_written_in_one_order_whatever_its_iteration_order() -> None:
    """1 and 9 share a hash slot, so the insertion order decides the iteration
    order of these equal frozensets; the written order must not follow it."""
    first, second = frozenset([1, 9]), frozenset([9, 1])
    assert first == second and list(first) != list(second)
    assert encode(first) == encode(second)


_ORDER_SCRIPT = (
    "import json; from lizyml.core.category_values import encode; "
    "print(json.dumps(encode(frozenset({'a', 'b', 'c', 'd', b'e', (1, 2), 3.5, 7}))))"
)


def test_a_frozenset_is_written_in_one_order_under_any_hash_seed() -> None:
    """Negative control (f): writing in hash order would differ across seeds."""
    outputs = {
        subprocess.run(
            [sys.executable, "-c", _ORDER_SCRIPT],
            env={"PYTHONHASHSEED": seed, "PATH": ""},
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        for seed in ("1", "2", "3")
    }
    assert len(outputs) == 1
