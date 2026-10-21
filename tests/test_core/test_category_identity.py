"""H-0138 acceptance 1: the structural category key (``category_identity_key``).

Numbers that are equal as numbers share a key (Python ``==``), strings and
bytes share one with their NumPy twins, and tuples / frozensets compare item
by item. Temporal values and values outside V have no key: their columns keep
today's pandas path (H-0138 decision 4).
"""

from __future__ import annotations

import datetime as dt
import decimal
import itertools
from typing import Any

import numpy as np
import pandas as pd
import pytest

from lizyml.core.category_temporal import CalendarDuration, calendar_units
from lizyml.core.category_values import (
    CATEGORY_IDENTITY_VERSION,
    category_identity_key,
    decode,
    encode,
    holds_temporal,
)
from tests.test_core.test_category_values import SAMPLES


def _wrap(value: Any, depth: int) -> Any:
    for _ in range(depth):
        value = (value,)
    return value


def _non_temporal() -> list[tuple[str, Any]]:
    return [
        (f"{t.__name__}[{i}]", v)
        for t, vs in SAMPLES.items()
        for i, v in enumerate(vs)
        if not holds_temporal(v)
    ]


NP_ONES = [
    t(1)
    for t in (np.bool_, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16,
              np.uint32, np.uint64, np.float32, np.float64, np.complex64, np.complex128)
]  # fmt: skip


def test_the_version_is_one() -> None:
    assert CATEGORY_IDENTITY_VERSION == 1


@pytest.mark.parametrize(("name", "value"), _non_temporal())
def test_every_non_temporal_value_has_a_hashable_key(name: str, value: Any) -> None:
    del name
    hash(category_identity_key(value))


@pytest.mark.parametrize(("name", "value"), _non_temporal())
def test_the_codec_round_trip_keeps_the_key(name: str, value: Any) -> None:
    del name
    assert category_identity_key(decode(encode(value))) == category_identity_key(value)


_TEMPORAL_VALUES = [v for vs in SAMPLES.values() for v in vs if holds_temporal(v)] + [
    (1, pd.Timestamp("2024-01-01")),
    frozenset({dt.date(2024, 1, 1)}),
]


@pytest.mark.parametrize(
    "value", [v for v in _TEMPORAL_VALUES if not calendar_units(v)], ids=repr
)
def test_a_temporal_value_has_a_key_that_survives_the_codec(value: Any) -> None:
    """H-0139 replaced H-0138's "a temporal value has no key"."""
    assert holds_temporal(value)
    assert category_identity_key(decode(encode(value))) == category_identity_key(value)


@pytest.mark.parametrize(
    "value",
    [v for v in _TEMPORAL_VALUES if calendar_units(v)]
    + [(1, np.timedelta64(2, "M")), frozenset({np.timedelta64(1, "Y")})],
    ids=repr,
)
def test_a_calendar_duration_has_no_key(value: Any) -> None:
    with pytest.raises(CalendarDuration):
        category_identity_key(value)


@pytest.mark.parametrize("value", [float("nan"), object(), np.float16(1), [1]])
def test_a_value_outside_v_has_no_key(value: Any) -> None:
    with pytest.raises(TypeError, match="outside V"):
        category_identity_key(value)


EQUAL = [
    (0, -0.0),
    (0, False),
    (0, decimal.Decimal("-0")),
    (-0.0, np.float64(-0.0)),
    (True, np.uint8(1)),
    (np.True_, np.uint8(1)),
    (1.5, decimal.Decimal("1.50")),
    (float("inf"), decimal.Decimal("Infinity")),
    (float("inf"), complex(float("inf"), 0)),
    (np.int64(-1), np.complex64(-1)),
    (np.uint16(2), np.complex128(2)),
    (2**70, decimal.Decimal(2**70)),
    ("x", np.str_("x")),
    (b"y", np.bytes_(b"y")),
    ((1, "a"), (True, np.str_("a"))),
    (frozenset({1, 2.5}), frozenset({np.int8(1), decimal.Decimal("2.5")})),
    *((decimal.Decimal("1"), one) for one in NP_ONES),  # #378
]
DISTINCT = [
    (np.float32(0.1), 0.1),
    (decimal.Decimal("0.1"), 0.1),
    (1, "1"),
    (b"a", "a"),
    (complex(1, -2), 1),
    (float("inf"), float("-inf")),
    *((one, _wrap(1, depth)) for one in NP_ONES for depth in (1, 2, 3, 8)),  # #380
    (2**70, np.True_),
    ((1,), frozenset({1})),
]


@pytest.mark.parametrize(("a", "b"), EQUAL, ids=repr)
def test_numerically_equal_values_share_a_key(a: Any, b: Any) -> None:
    assert category_identity_key(a) == category_identity_key(b)


@pytest.mark.parametrize(("a", "b"), DISTINCT, ids=repr)
def test_different_values_have_different_keys(a: Any, b: Any) -> None:
    assert category_identity_key(a) != category_identity_key(b)


def _python_equal(a: Any, b: Any) -> bool | None:
    """``a == b`` when both directions answer the same plain bool, else None."""
    try:
        forward, backward = a == b, b == a
    except Exception:  # noqa: BLE001 -- #378: one direction raises
        return None
    if not isinstance(forward, (bool, np.bool_)) or not isinstance(
        backward, (bool, np.bool_)
    ):
        return None
    return bool(forward) if bool(forward) == bool(backward) else None


def test_the_key_agrees_with_python_equality_over_every_sample_pair() -> None:
    """Where Python's == answers one plain bool both ways, the key agrees.

    The exceptions are the pairs the decided contract settles differently: a
    NumPy scalar beside a container (NumPy broadcasts, the contract compares
    structurally) is the only kind measured (Codex run h0138-mechanism-design).
    """
    disagreements = []
    for (_, a), (_, b) in itertools.combinations(_non_temporal(), 2):
        equal = _python_equal(a, b)
        if equal is None:
            continue
        if (category_identity_key(a) == category_identity_key(b)) != equal:
            disagreements.append((a, b))
    allowed = [
        (a, b)
        for a, b in disagreements
        if isinstance(a, np.generic) != isinstance(b, np.generic)
        and (type(a) in (tuple, frozenset) or type(b) in (tuple, frozenset))
    ]
    assert disagreements == allowed
