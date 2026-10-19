"""H-0133 P1 reference predicate: is a learned category value in the closed set V?

A standalone copy of the Proposal's rule, used only by the firing probe, so
the census applies the proposed predicate itself (Proposal review round 1).
Types are matched exactly with ``is``; no method of a value outside the
accepted types runs. ``reason(value)`` returns ``None`` for a value in V and
otherwise a short reason.
"""

from __future__ import annotations

import datetime as dt
import decimal
import inspect
import math
import zoneinfo
from typing import Any

import numpy as np
import pandas as pd

MAX_DEPTH = 8

PLAIN = (str, int, float, bool)
EXTENDED = (bytes, decimal.Decimal, dt.date, dt.datetime, dt.timedelta, complex, pd.Timestamp, pd.Timedelta)
NUMPY = (
    np.bool_, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64,
    np.float32, np.float64, np.complex64, np.complex128, np.str_, np.bytes_, np.datetime64, np.timedelta64,
)
CONTAINERS = (tuple, frozenset)


def _is(value: Any, types: tuple[type, ...]) -> bool:
    t = type(value)
    return any(t is k for k in types)


def _tz_reason(tz: Any) -> str | None:
    if tz is None:
        return None
    if type(tz) is dt.timezone:
        return None
    if type(tz) is zoneinfo.ZoneInfo:
        return None if tz.key is not None else "ZoneInfo without a key"
    try:
        import pytz
    except ImportError:  # pragma: no cover - pandas 3 without pytz
        return f"tzinfo {type(tz).__name__}"
    # P1: read the name without running a descriptor of a caller's class.
    zone = inspect.getattr_static(tz, "zone", None)
    if isinstance(zone, str) and zone in pytz.all_timezones_set:
        base = pytz.timezone(zone)
        family = getattr(base, "_tzinfos", {}) or {}
        if tz is base or any(tz is v for v in family.values()):
            return None
    return f"tzinfo {type(tz).__name__}"


def _scalar_reason(value: Any) -> str | None:
    if _is(value, (float, np.float32, np.float64)) and math.isnan(float(value)):
        return "NaN"
    if type(value) is decimal.Decimal and value.is_nan():
        return "Decimal NaN"
    if _is(value, (complex, np.complex64, np.complex128)) and (
        math.isnan(value.real) or math.isnan(value.imag)
    ):
        return "complex NaN"
    if _is(value, (np.datetime64, np.timedelta64)) and np.isnat(value):
        return "NaT"
    if _is(value, (dt.datetime, pd.Timestamp)):
        return _tz_reason(value.tzinfo)
    return None


def reason(value: Any, depth: int = 0) -> str | None:
    if _is(value, CONTAINERS):
        if depth >= MAX_DEPTH:
            return f"nesting deeper than {MAX_DEPTH}"
        for item in value:
            r = reason(item, depth + 1)
            if r is not None:
                return r
        return None
    if _is(value, PLAIN) or _is(value, EXTENDED) or _is(value, NUMPY):
        return _scalar_reason(value)
    return f"type {type(value).__name__}"
