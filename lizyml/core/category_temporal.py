"""category_temporal -- the identity key of temporal category values (H-0139).

Two temporal values are one category exactly when their keys are equal:

- an aware ``datetime`` / ``Timestamp`` by its UTC instant, whatever the time
  zone's provider (UTC, IANA, pytz, fixed offset) or fold label;
- a naive ``datetime`` / ``Timestamp`` / ``np.datetime64`` by its wall-clock
  coordinate, exactly (sub-nanosecond units included);
- a ``date`` by its ordinal; naive, aware and ``date`` keys never meet;
- a fixed duration (``pd.Timedelta``, ``datetime.timedelta``, fixed-unit
  ``np.timedelta64``) by its exact length; a unitless ``np.timedelta64`` by
  its raw count, apart from every fixed duration.

Coordinates are attoseconds held as Python ``int`` (no int64 or nanosecond
range to overflow). A ``np.timedelta64`` in ``Y`` or ``M`` units has no
length and no key (:class:`CalendarDuration`): fit refuses such a column
(H-0139 decision 4).
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import numpy as np
import pandas as pd

#: Attoseconds per fixed NumPy / pandas time unit.
_ATTOSECONDS: dict[str, int] = {
    "W": 604800 * 10**18,
    "D": 86400 * 10**18,
    "h": 3600 * 10**18,
    "m": 60 * 10**18,
    "s": 10**18,
    "ms": 10**15,
    "us": 10**12,
    "ns": 10**9,
    "ps": 10**6,
    "fs": 10**3,
    "as": 1,
}
#: ``np.timedelta64`` units without a fixed length.
CALENDAR_UNITS = ("Y", "M")
_EPOCH = dt.datetime(1970, 1, 1)
_US = dt.timedelta(microseconds=1)


class CalendarDuration(TypeError):
    """A ``np.timedelta64`` in ``Y`` or ``M`` units: it has no identity key."""


def calendar_units(value: Any) -> set[str]:
    """The ``Y`` / ``M`` units of the ``np.timedelta64`` values *value* is or
    holds (tuples and frozensets at any depth); empty when there are none."""
    if type(value) is tuple or type(value) is frozenset:
        found: set[str] = set()
        for item in value:
            found |= calendar_units(item)
        return found
    if type(value) is np.timedelta64:
        unit = np.datetime_data(value.dtype)[0]
        if unit in CALENDAR_UNITS:
            return {unit}
    return set()


def _days_from_civil(year: int, month: int, day: int) -> int:
    """Days from 1970-01-01 in the proleptic Gregorian calendar (exact ints)."""
    year -= month <= 2
    era = year // 400
    year_of_era = year - era * 400
    day_of_year = (153 * (month + (-3 if month > 2 else 9)) + 2) // 5 + day - 1
    day_of_era = year_of_era * 365 + year_of_era // 4 - year_of_era // 100 + day_of_year
    return era * 146097 + day_of_era - 719468


def _raw(value: Any) -> int:
    return int(value.astype(np.int64))


def _datetime64_coordinate(value: Any) -> Any:
    unit, multiple = np.datetime_data(value.dtype)
    count = _raw(value) * int(multiple)
    if unit == "Y":
        return _days_from_civil(1970 + count, 1, 1) * _ATTOSECONDS["D"]
    if unit == "M":
        return (
            _days_from_civil(1970 + count // 12, count % 12 + 1, 1) * _ATTOSECONDS["D"]
        )
    if unit == "generic":
        return ("datetime64-generic", count)
    return count * _ATTOSECONDS[unit]


def temporal_key(value: Any) -> tuple[Any, ...]:
    """The key of one temporal scalar of V (exact-type dispatch).

    Raises:
        CalendarDuration: for a ``np.timedelta64`` in ``Y`` or ``M`` units.
        TypeError: for a value that is not a temporal scalar.
    """
    kind = type(value)
    if kind is pd.Timestamp:
        # asm8 is UTC for an aware Timestamp: no offset to subtract again.
        coordinate = _raw(value.asm8) * _ATTOSECONDS[value.unit]
        return ("datetime", "naive" if value.tzinfo is None else "aware", coordinate)
    if kind is dt.datetime:
        wall_us = (value.replace(tzinfo=None) - _EPOCH) // _US
        if value.tzinfo is None:
            return ("datetime", "naive", wall_us * _ATTOSECONDS["us"])
        offset_us = value.utcoffset() // _US  # fold and provider offsets enter here
        return ("datetime", "aware", (wall_us - offset_us) * _ATTOSECONDS["us"])
    if kind is dt.date:
        return ("date", value.toordinal())
    if kind is np.datetime64:
        return ("datetime", "naive", _datetime64_coordinate(value))
    if kind is pd.Timedelta:
        return ("duration", _raw(value.asm8) * _ATTOSECONDS[value.unit])
    if kind is dt.timedelta:
        microseconds = (
            value.days * 86400 + value.seconds
        ) * 1_000_000 + value.microseconds
        return ("duration", microseconds * _ATTOSECONDS["us"])
    if kind is np.timedelta64:
        unit, multiple = np.datetime_data(value.dtype)
        if unit in CALENDAR_UNITS:
            raise CalendarDuration(
                f"np.timedelta64 in {unit} units has no identity key"
            )
        if unit == "generic":
            return ("timedelta64-generic", _raw(value))
        return ("duration", _raw(value) * int(multiple) * _ATTOSECONDS[unit])
    raise TypeError(f"not a temporal value: {kind.__name__}")
