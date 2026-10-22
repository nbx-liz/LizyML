"""H-0139 prototype: the temporal identity key and the path a column would take.

``category_key`` is the decided contract (registry card D-20261022-002, the
key of Codex run ``h0139-mechanism-design``): aware datetime / Timestamp
values by UTC instant, naive datetime / Timestamp / datetime64 values by
wall-clock coordinate, both in attoseconds; ``date`` by ordinal; fixed
durations by exact length; unitless ``timedelta64`` by an opaque raw key;
``timedelta64`` in ``Y`` / ``M`` units has no key. Non-temporal values use
H-0138's ``category_identity_key`` unchanged; tuples and frozensets recurse.

``classify`` runs pandas' cast on one column and reports the path:

- ``ym_refused``: a row value or a declared category holds a Y/M duration;
- ``outside_v``: a value is outside H-0133's V -> today's pandas path;
- ``faithful``: the cast succeeds and agrees with the key -> today's result;
- ``cast_raised`` / ``predicate_false``: built by key (stand-in).

The H-0139 activation is a temporal column taking ``cast_raised`` or
``predicate_false``, or any column taking ``ym_refused``.
"""

from __future__ import annotations

import datetime as dt
import math
from typing import Any

import numpy as np
import pandas as pd

from lizyml.core.category_values import category_identity_key, refusal

_AS = {
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
_EPOCH = dt.datetime(1970, 1, 1)
_US = dt.timedelta(microseconds=1)
_TEMPORAL = (dt.date, dt.timedelta, np.datetime64, np.timedelta64)


class CalendarDuration(TypeError):
    """A timedelta64 in Y or M units: no key (refused at fit)."""


def is_temporal(value: Any) -> bool:
    if type(value) in (tuple, frozenset):
        return any(is_temporal(v) for v in value)
    return isinstance(value, _TEMPORAL)


def calendar_units(value: Any) -> set[str]:
    if type(value) in (tuple, frozenset):
        return set().union(*(calendar_units(v) for v in value)) if value else set()
    if type(value) is np.timedelta64:
        unit = np.datetime_data(value.dtype)[0]
        if unit in ("Y", "M"):
            return {unit}
    return set()


def _days_from_civil(y: int, m: int, d: int) -> int:
    y -= m <= 2
    era = y // 400
    yoe = y - era * 400
    doy = (153 * (m + (-3 if m > 2 else 9)) + 2) // 5 + d - 1
    doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
    return era * 146097 + doe - 719468


def _datetime64_coordinate(value: Any) -> Any:
    unit, mult = np.datetime_data(value.dtype)
    n = int(value.view("i8")) * int(mult)
    if unit == "Y":
        return _days_from_civil(1970 + n, 1, 1) * _AS["D"]
    if unit == "M":
        return _days_from_civil(1970 + n // 12, n % 12 + 1, 1) * _AS["D"]
    if unit == "generic":
        return ("datetime64-generic", n)
    return n * _AS[unit]


def _temporal_key(value: Any) -> Any:
    kind = type(value)
    if kind is pd.Timestamp:
        coord = int(value.asm8.view("i8")) * _AS[value.unit]
        return ("datetime", "naive" if value.tzinfo is None else "aware", coord)
    if kind is dt.datetime:
        wall_us = (value.replace(tzinfo=None) - _EPOCH) // _US
        if value.tzinfo is None:
            return ("datetime", "naive", wall_us * 10**12)
        offset_us = value.utcoffset() // _US
        return ("datetime", "aware", (wall_us - offset_us) * 10**12)
    if kind is dt.date:
        return ("date", value.toordinal())
    if kind is np.datetime64:
        return ("datetime", "naive", _datetime64_coordinate(value))
    if kind is pd.Timedelta:
        return ("duration", int(value.asm8.view("i8")) * _AS[value.unit])
    if kind is dt.timedelta:
        us = (value.days * 86400 + value.seconds) * 1_000_000 + value.microseconds
        return ("duration", us * 10**12)
    if kind is np.timedelta64:
        unit, mult = np.datetime_data(value.dtype)
        if unit in ("Y", "M"):
            raise CalendarDuration(f"timedelta64 in {unit}")
        raw = int(value.view("i8"))
        if unit == "generic":
            return ("timedelta64-generic", raw)
        return ("duration", raw * int(mult) * _AS[unit])
    raise TypeError(f"no temporal key for {kind.__name__}")


def category_key(value: Any) -> Any:
    if type(value) is tuple:
        return ("tuple", tuple(category_key(v) for v in value))
    if type(value) is frozenset:
        return ("frozenset", frozenset(category_key(v) for v in value))
    if isinstance(value, _TEMPORAL):
        return _temporal_key(value)
    return category_identity_key(value)


def _missing(value: Any) -> bool:
    if value is None or value is pd.NA or value is pd.NaT:
        return True
    return isinstance(value, (float, np.floating)) and math.isnan(value)


def faithful(original: pd.Series, cast: pd.Series) -> bool:
    cats = list(cast.cat.categories)
    keys = [category_key(c) for c in cats]
    if len(set(keys)) != len(keys):
        return False
    codes = cast.cat.codes.to_numpy()
    for raw, code in zip(original.to_numpy(dtype=object), codes, strict=True):
        if _missing(raw):
            if code != -1:
                return False
            continue
        if code < 0 or keys[code] != category_key(raw):
            return False
    return True


def dtype_kind(series: pd.Series) -> str:
    if isinstance(series.dtype, pd.CategoricalDtype):
        return "category"
    if series.dtype == object:
        return "object"
    if pd.api.types.is_string_dtype(series):
        return "string"
    return "other:" + str(series.dtype)


def classify(series: pd.Series) -> dict[str, Any]:
    """The path of one column of any dtype, with the facts it was read from."""
    values = [v for v in series.to_numpy(dtype=object) if not _missing(v)]
    declared: list[Any] = []
    if isinstance(series.dtype, pd.CategoricalDtype):
        declared = list(series.cat.categories)
    every = values + declared
    temporal = any(is_temporal(v) for v in every)
    units = sorted(set().union(*(calendar_units(v) for v in every)) if every else set())
    out: dict[str, Any] = {"temporal": temporal, "ym_units": units}
    if units:
        out["path"] = "ym_refused"
    elif any(refusal(v) is not None for v in every):
        out["path"] = "outside_v"
    else:
        try:
            cast = series.astype("category")
        except Exception:  # noqa: BLE001 -- the outcome being classified
            out["path"] = "cast_raised"
            return out
        if declared and list(cast.cat.categories) != declared:
            out["path"] = "predicate_false"
        else:
            out["path"] = "faithful" if faithful(series, cast) else "predicate_false"
    return out
