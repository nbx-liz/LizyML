"""category_values -- the closed set of saveable category values and its codec (H-0133).

``Model.export``, ``lizyml.persistence.export`` and ``export_code`` accept
feature-category values from the set V, and only in states that answer the
same after joblib and after this codec (H-0136, one check in
``persistence.exporter.check_category_values``). A value is in V when its type is one
of :data:`ADMITTED_TYPES` exactly (``type(v) is T``: a subclass is refused, and
its code never runs), it equals itself (NaN, ``Decimal`` NaN, a NaN complex and
NaT are refused at any depth), a ``tuple`` / ``frozenset`` nests at most
:data:`MAX_DEPTH` containers, and a timezone is one this module can restore
with an equal value and hash.

:func:`encode` writes a value of V as JSON: plain values (and numpy scalars of
them) as themselves, everything else as a tagged object
``{"$lizyml": <tag>, ...}`` with a fixed field set. :func:`decode` reads only
those shapes back, refusing anything else. The generated ``train.py`` /
``predict.py`` carry a copy of the decoder (``codegen/templates.py``); a test
holds the two to the same answers over the whole population.

The bound is the values: a caller that replaces numpy, pandas or the standard
library inside the process is outside it (as in H-0095).
"""

from __future__ import annotations

import base64
import datetime as dt
import decimal
import inspect
import json
import math
import zoneinfo
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from lizyml.core.category_temporal import (
    calendar_units,
    holds_unitless_datetime,
    temporal_key,
)
from lizyml.core.exceptions import ErrorCode, LizyMLError

#: The deepest ``tuple`` / ``frozenset`` nesting V admits (``((1,),)`` is 2).
MAX_DEPTH = 8

_PLAIN: tuple[type, ...] = (str, int, float, bool)
#: numpy scalars written as their ``.item()`` (a plain value equal to them).
_NUMPY_PLAIN: tuple[type, ...] = (
    np.bool_,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.float32,
    np.float64,
    np.str_,
)
_NUMPY_COMPLEX: tuple[type, ...] = (np.complex64, np.complex128)
_CONTAINERS: tuple[type, ...] = (tuple, frozenset)

#: Every type a value of V can have (H-0133 P1). Closed: a value of any other
#: type -- a subclass included -- is outside V.
ADMITTED_TYPES: tuple[type, ...] = (
    *_PLAIN,
    bytes,
    decimal.Decimal,
    dt.date,
    dt.datetime,
    dt.timedelta,
    complex,
    pd.Timestamp,
    pd.Timedelta,
    *_NUMPY_PLAIN,
    *_NUMPY_COMPLEX,
    np.bytes_,
    np.datetime64,
    np.timedelta64,
    *_CONTAINERS,
)

#: The constructors, untyped: the unit is read from the saved form, so numpy's
#: literal-unit overloads cannot apply (a wrong unit raises at decode).
_DATETIME64: Any = np.datetime64
_TIMEDELTA64: Any = np.timedelta64

_TAG = "$lizyml"
_EPOCH = dt.datetime(1970, 1, 1)
_US = dt.timedelta(microseconds=1)
_TIMESTAMP_UNITS = ("s", "ms", "us", "ns")


def _is(value: Any, types: tuple[type, ...]) -> bool:
    kind = type(value)
    return any(kind is t for t in types)


def _pytz_zone(tz: Any) -> str | None:
    """The zone name of a pytz zone instance, or ``None`` for anything else.

    The name is read with ``inspect.getattr_static`` (no descriptor of a
    caller's class runs) and the instance must be pytz's own for that name.
    """
    try:
        import pytz  # type: ignore[import-untyped, unused-ignore]
    except ImportError:
        return None
    zone = inspect.getattr_static(tz, "zone", None)
    if type(zone) is not str or zone not in pytz.all_timezones_set:
        return None
    base = pytz.timezone(zone)
    family = inspect.getattr_static(base, "_tzinfos", None)
    members = family.values() if type(family) is dict else ()
    if tz is base or any(tz is m for m in members):
        return zone
    return None


def _timestamp_tz_refusal(tz: Any) -> str | None:
    if tz is None or type(tz) is dt.timezone:
        return None
    if type(tz) is zoneinfo.ZoneInfo:
        return None if tz.key is not None else "a ZoneInfo without a key"
    if _pytz_zone(tz) is not None:
        return None
    return f"the tzinfo type {type(tz).__name__}"


def _datetime_tz_refusal(tz: Any) -> str | None:
    # Narrower than Timestamp: a datetime compares across tzinfo instances in
    # UTC and calls a DST-fold time unequal (PEP 495), so only a fixed offset
    # or the cached ZoneInfo instance restores to an equal value.
    if tz is None or type(tz) is dt.timezone:
        return None
    if (
        type(tz) is zoneinfo.ZoneInfo
        and tz.key is not None
        and zoneinfo.ZoneInfo(tz.key) is tz
    ):
        return None
    return (
        f"the datetime tzinfo {type(tz).__name__} "
        "(only fixed offsets and cached ZoneInfo)"
    )


def _scalar_refusal(value: Any) -> str | None:
    if _is(value, (float, np.float32, np.float64)) and math.isnan(float(value)):
        return "a NaN"
    if type(value) is decimal.Decimal and value.is_nan():
        return "a Decimal NaN"
    if _is(value, (complex, *_NUMPY_COMPLEX)):
        c = complex(value)
        if math.isnan(c.real) or math.isnan(c.imag):
            return "a complex NaN"
    if _is(value, (np.datetime64, np.timedelta64)) and np.isnat(value):
        return "NaT"
    if type(value) is pd.Timestamp:
        return _timestamp_tz_refusal(value.tzinfo)
    if type(value) is dt.datetime:
        return _datetime_tz_refusal(value.tzinfo)
    return None


def _refusal(value: Any, depth: int) -> str | None:
    if _is(value, _CONTAINERS):
        if depth >= MAX_DEPTH:
            return f"a tuple / frozenset nested deeper than {MAX_DEPTH}"
        for item in value:
            reason = _refusal(item, depth + 1)
            if reason is not None:
                return reason
        return None
    if not _is(value, ADMITTED_TYPES):
        return f"the type {type(value).__name__}"
    return _scalar_refusal(value)


def refusal(value: Any) -> str | None:
    """``None`` when *value* is in V, else why it is not (no caller code runs)."""
    return _refusal(value, 0)


def type_name(value: Any) -> str:
    """The name of *value*'s type, never raising."""
    try:
        name = type.__getattribute__(type(value), "__name__")
    except Exception:  # noqa: BLE001 - a hostile metaclass must not stop fit
        return "<unknown type>"
    return name if isinstance(name, str) else "<unknown type>"


def shown(value: Any) -> str:
    """``repr(value)``, or ``<unprintable TYPE>`` when ``repr`` raises."""
    try:
        text = repr(value)
    except Exception:  # noqa: BLE001 - a hostile __repr__ must not stop fit
        return f"<unprintable {type_name(value)}>"
    return text if isinstance(text, str) else f"<unprintable {type_name(value)}>"


# ---------------------------------------------------------------------------
# encoding
# ---------------------------------------------------------------------------


def _tz_shape(value: Any) -> dict[str, Any] | None:
    tz = value.tzinfo
    if tz is None:
        return None
    if type(tz) is zoneinfo.ZoneInfo:
        return {"kind": "iana", "name": tz.key}
    # datetime.timezone, and a pytz zone (written as the value's own offset:
    # restored through ZoneInfo its hash would differ at fold 1).
    offset = value.utcoffset()
    return {"kind": "offset", "microseconds": offset // _US}


def _encode(value: Any) -> Any:
    if _is(value, _PLAIN):
        return value
    if _is(value, _NUMPY_PLAIN):
        return value.item()
    if _is(value, (bytes, np.bytes_)):
        return {_TAG: "bytes", "b64": base64.b64encode(bytes(value)).decode("ascii")}
    if type(value) is decimal.Decimal:
        return {_TAG: "decimal", "str": str(value)}
    if _is(value, (complex, *_NUMPY_COMPLEX)):
        c = complex(value)
        return {_TAG: "complex", "real": c.real, "imag": c.imag}
    if type(value) is pd.Timestamp:
        return {
            _TAG: "timestamp",
            "unit": value.unit,
            "value": int(value.asm8.astype(np.int64)),
            "tz": _tz_shape(value),
        }
    if type(value) is dt.datetime:
        wall = value.replace(tzinfo=None)
        return {
            _TAG: "datetime",
            "us": (wall - _EPOCH) // _US,
            "fold": value.fold,
            "tz": _tz_shape(value),
        }
    if type(value) is dt.date:
        return {_TAG: "date", "iso": value.isoformat()}
    if _is(value, (np.datetime64, np.timedelta64)):
        unit, count = np.datetime_data(value.dtype)
        tag = "datetime64" if type(value) is np.datetime64 else "timedelta64"
        return {
            _TAG: tag,
            "unit": unit,
            "count": count,
            "value": int(value.astype(np.int64)),
        }
    if type(value) is pd.Timedelta:
        return {
            _TAG: "pd_timedelta",
            "unit": value.unit,
            "value": int(value.asm8.astype(np.int64)),
        }
    if type(value) is dt.timedelta:
        return {
            _TAG: "timedelta",
            "days": value.days,
            "seconds": value.seconds,
            "microseconds": value.microseconds,
        }
    if type(value) is tuple:
        return {_TAG: "tuple", "items": [_encode(v) for v in value]}
    if type(value) is frozenset:
        items = [_encode(v) for v in value]
        items.sort(key=lambda item: json.dumps(item, sort_keys=True))
        return {_TAG: "frozenset", "items": items}
    raise AssertionError(f"unreachable: {type(value).__name__}")  # pragma: no cover


def encode(value: Any) -> Any:
    """*value* as the JSON value the saved form holds.

    Raises:
        LizyMLError: ``SERIALIZATION_FAILED`` when *value* is outside V, with
            the reason in ``context["reason"]``.
    """
    reason = refusal(value)
    if reason is not None:
        raise LizyMLError(
            code=ErrorCode.SERIALIZATION_FAILED,
            user_message=f"The category value cannot be saved: it is {reason}.",
            context={"type": type(value).__name__, "reason": reason},
        )
    return _encode(value)


# ---------------------------------------------------------------------------
# decoding
# ---------------------------------------------------------------------------

_FIELDS: dict[str, frozenset[str]] = {
    "bytes": frozenset({"b64"}),
    "decimal": frozenset({"str"}),
    "complex": frozenset({"real", "imag"}),
    "date": frozenset({"iso"}),
    "datetime": frozenset({"us", "fold", "tz"}),
    "timestamp": frozenset({"unit", "value", "tz"}),
    "datetime64": frozenset({"unit", "count", "value"}),
    "timedelta64": frozenset({"unit", "count", "value"}),
    "pd_timedelta": frozenset({"unit", "value"}),
    "timedelta": frozenset({"days", "seconds", "microseconds"}),
    "tuple": frozenset({"items"}),
    "frozenset": frozenset({"items"}),
}


class _Malformed(ValueError):
    pass


def _int(obj: Any, field: str) -> int:
    value = obj[field]
    if type(value) is not int:
        raise _Malformed(f"field {field!r} is not an integer")
    return value


def _str(obj: Any, field: str) -> str:
    value = obj[field]
    if type(value) is not str:
        raise _Malformed(f"field {field!r} is not a string")
    return value


def _float(obj: Any, field: str) -> float:
    value = obj[field]
    if type(value) is not float:
        raise _Malformed(f"field {field!r} is not a float")
    return value


def _tz(shape: Any) -> dt.tzinfo | None:
    if shape is None:
        return None
    if type(shape) is not dict:
        raise _Malformed("the timezone is not an object")
    if set(shape) == {"kind", "name"} and shape["kind"] == "iana":
        try:
            return zoneinfo.ZoneInfo(_str(shape, "name"))
        except (zoneinfo.ZoneInfoNotFoundError, ValueError) as exc:
            raise _Malformed(f"unknown timezone {shape['name']!r}") from exc
    if set(shape) == {"kind", "microseconds"} and shape["kind"] == "offset":
        try:
            return dt.timezone(dt.timedelta(microseconds=_int(shape, "microseconds")))
        except ValueError as exc:
            raise _Malformed("the timezone offset is out of range") from exc
    raise _Malformed("the timezone has an unknown shape")


def _units(obj: Any) -> str:
    unit = _str(obj, "unit")
    if unit not in _TIMESTAMP_UNITS:
        raise _Malformed(f"unit {unit!r} is not one of {_TIMESTAMP_UNITS}")
    return unit


def _decode_tagged(obj: dict[str, Any], depth: int) -> Any:
    tag = obj[_TAG]
    if type(tag) is not str or tag not in _FIELDS:
        raise _Malformed(f"unknown tag {tag!r}")
    if set(obj) - {_TAG} != _FIELDS[tag]:
        raise _Malformed(f"tag {tag!r} needs exactly the fields {sorted(_FIELDS[tag])}")
    if tag == "bytes":
        return base64.b64decode(_str(obj, "b64").encode("ascii"), validate=True)
    if tag == "decimal":
        return decimal.Decimal(_str(obj, "str"))
    if tag == "complex":
        return complex(_float(obj, "real"), _float(obj, "imag"))
    if tag == "date":
        return dt.date.fromisoformat(_str(obj, "iso"))
    if tag == "datetime":
        fold = _int(obj, "fold")
        if fold not in (0, 1):
            raise _Malformed("fold is not 0 or 1")
        wall = _EPOCH + dt.timedelta(microseconds=_int(obj, "us"))
        return wall.replace(tzinfo=_tz(obj["tz"]), fold=fold)
    if tag == "timestamp":
        unit = _units(obj)
        stamp = pd.Timestamp(_DATETIME64(_int(obj, "value"), unit))
        tz = _tz(obj["tz"])
        return stamp if tz is None else stamp.tz_localize("UTC").tz_convert(tz)
    if tag in ("datetime64", "timedelta64"):
        unit, count = _str(obj, "unit"), _int(obj, "count")
        if count < 1:
            raise _Malformed("count is not positive")
        if unit == "generic" and count == 1:
            # NumPy builds no unitless datetime64 from an int; the bytes are
            # the count itself (H-0139: the key must survive the codec).
            return np.array(_int(obj, "value"), dtype=np.int64).view(tag)[()]
        kind = _DATETIME64 if tag == "datetime64" else _TIMEDELTA64
        return kind(_int(obj, "value"), unit if count == 1 else f"{count}{unit}")
    if tag == "pd_timedelta":
        return pd.Timedelta(_TIMEDELTA64(_int(obj, "value"), _units(obj)))
    if tag == "timedelta":
        return dt.timedelta(
            days=_int(obj, "days"),
            seconds=_int(obj, "seconds"),
            microseconds=_int(obj, "microseconds"),
        )
    items = obj["items"]
    if type(items) is not list:
        raise _Malformed("items is not a list")
    if depth >= MAX_DEPTH:
        raise _Malformed(f"nested deeper than {MAX_DEPTH}")
    decoded = [_decode(item, depth + 1) for item in items]
    return tuple(decoded) if tag == "tuple" else frozenset(decoded)


def _decode(obj: Any, depth: int) -> Any:
    if _is(obj, _PLAIN):
        return obj
    if type(obj) is dict and _TAG in obj:
        return _decode_tagged(obj, depth)
    raise _Malformed(f"a {type(obj).__name__} is not a category value")


def decode(obj: Any) -> Any:
    """The value :func:`encode` wrote as *obj*.

    Raises:
        LizyMLError: ``DESERIALIZATION_FAILED`` when *obj* is not one of the
            shapes :func:`encode` writes.
    """
    try:
        value = _decode(obj, 0)
        # A shape encode never writes (JSON NaN, Decimal("NaN"), NaT, ...)
        # decodes to a value outside V: refuse it rather than return it.
        reason = refusal(value)
        if reason is not None:
            raise _Malformed(f"it decodes to {reason}, which encode never writes")
        # Only the canonical form encode writes is read: a timedelta of 0 days
        # and 86400 seconds, or a frozenset with a repeated item, decodes but
        # is not what encode writes for its value.
        if _encode(value) != obj:
            raise _Malformed("it is not the form encode writes for its value")
        return value
    except (
        _Malformed,
        ValueError,
        TypeError,
        OverflowError,
        decimal.InvalidOperation,
    ) as exc:
        raise LizyMLError(
            code=ErrorCode.DESERIALIZATION_FAILED,
            user_message=f"A saved category value is malformed: {exc}",
            context={"reason": str(exc)},
            cause=exc,
        ) from exc


# ---------------------------------------------------------------------------
# identity (H-0138)
# ---------------------------------------------------------------------------

#: The identity a fitted encoder state records (``category_identity_version``).
#: H-0139 widened version 1 to temporal values (H-0138 was never released).
CATEGORY_IDENTITY_VERSION = 1

#: ``DataFrame.attrs`` key naming the categorical columns the builder keeps
#: as object because pandas' cast raised or was not faithful (H-0138). It
#: names columns only; their categories are learned inside each fit.
IDENTITY_OBJECT_COLUMNS = "lizyml_identity_object_columns"

_TEMPORAL: tuple[type, ...] = (
    dt.date,
    dt.datetime,
    dt.timedelta,
    pd.Timestamp,
    pd.Timedelta,
    np.datetime64,
    np.timedelta64,
)
_INTEGERS: tuple[type, ...] = (
    bool,
    int,
    np.bool_,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
)
_FLOATS: tuple[type, ...] = (float, np.float32, np.float64)


def holds_temporal(value: Any) -> bool:
    """Whether *value* is, or holds at any depth, a temporal value.

    Observational only since H-0139 (tests and the census): temporal values
    have a key and no longer route a column to the pandas path.
    """
    if _is(value, _CONTAINERS):
        return any(holds_temporal(item) for item in value)
    return isinstance(value, _TEMPORAL)


def _number_key(value: Any) -> tuple[Any, ...] | None:
    if _is(value, _INTEGERS):
        return ("number", "finite", int(value), 1)
    if _is(value, _FLOATS):
        x = float(value)
        if math.isinf(x):
            return ("number", "infinity", 1 if x > 0 else -1)
        n, d = x.as_integer_ratio()
        return ("number", "finite", n, d)
    if type(value) is decimal.Decimal:
        if value.is_infinite():
            return ("number", "infinity", -1 if value.is_signed() else 1)
        n, d = value.as_integer_ratio()
        return ("number", "finite", n, d)
    if _is(value, (complex, *_NUMPY_COMPLEX)):
        c = complex(value)
        real = _number_key(c.real)
        if c.imag == 0:
            return real
        return ("number", "complex", real, _number_key(c.imag))
    return None


def category_identity_key(value: Any) -> Any:
    """The structural key of a value of V (H-0138, temporal values H-0139).

    Two values are one category exactly when their keys are equal: numbers
    equal as numbers (``-0.0``, ``0``, ``False``, ``Decimal("-0")``), a string
    or bytes value and its NumPy twin, temporal values by instant, wall-clock
    coordinate or length (:mod:`lizyml.core.category_temporal`), and tuples /
    frozensets item by item. The key never runs a value's own ``==`` or
    ``hash``.

    Raises:
        TypeError: for a value outside V (its column keeps today's pandas
            path) or one holding a ``np.timedelta64`` in ``Y`` / ``M`` units
            (``CalendarDuration``; fit refuses such a column).
    """
    if refusal(value) is not None:
        raise TypeError(
            f"no category identity key for a value outside V ({type_name(value)})"
        )
    return _key(value)


def check_keyless_temporals(series: pd.Series, column: Any) -> None:
    """Refuse a column holding a temporal value with no identity key: a
    ``np.timedelta64`` in ``Y`` / ``M`` units (H-0139 decision 4) or a
    unitless ``np.datetime64`` (the owner's ruling after the implementation
    review's round 2) -- in a row, inside a tuple / frozenset, or in a
    declared category, used or not. Run before fit learns the column.

    Raises:
        LizyMLError: ``DATA_SCHEMA_INVALID``, naming the column (and the
            units of a calendar duration).
    """
    values = _values(series)
    units: set[str] = set()
    for value in values:
        units |= calendar_units(value)
    if not units and any(holds_unitless_datetime(v) for v in values):
        raise LizyMLError(
            ErrorCode.DATA_SCHEMA_INVALID,
            user_message=(
                f"Column '{column}' contains unitless numpy datetime64 categories."
            ),
            context={"column": column, "reason": "unitless_datetime64"},
        )
    if units:
        shown = sorted(units)
        raise LizyMLError(
            ErrorCode.DATA_SCHEMA_INVALID,
            user_message=(
                f"Column '{column}' contains calendar-unit numpy timedelta64 "
                f"categories: {shown}."
            ),
            context={
                "column": column,
                "units": shown,
                "reason": "calendar_timedelta64",
            },
        )


def identity_fields(state: Mapping[str, Any]) -> tuple[int | None, list[Any]]:
    """``(version, identity_columns)`` of an encoder state (H-0138).

    ``(None, [])`` for a state saved before H-0138: it keeps the pandas lookup.

    Raises:
        LizyMLError: ``DESERIALIZATION_FAILED`` for an identity version other
            than ``1`` (``True`` and ``1.0`` included) or a malformed
            ``identity_columns``.
    """
    if "category_identity_version" not in state:
        return None, []
    version = state["category_identity_version"]
    columns = state.get("identity_columns")
    categories = state.get("categories")
    if type(version) is not int or version != CATEGORY_IDENTITY_VERSION:
        raise LizyMLError(
            ErrorCode.DESERIALIZATION_FAILED,
            user_message=(
                f"The saved encoder state names category identity version "
                f"{version!r}; this LizyML reads only "
                f"{CATEGORY_IDENTITY_VERSION}."
            ),
            context={"category_identity_version": repr(version)},
        )
    if (
        type(columns) is not list
        or not isinstance(categories, Mapping)
        or not all(_names_a_column(c, categories) for c in columns)
    ):
        raise LizyMLError(
            ErrorCode.DESERIALIZATION_FAILED,
            user_message="The saved encoder state has a malformed identity_columns.",
            context={"identity_columns": repr(columns)[:200]},
        )
    return version, list(columns)


def _names_a_column(column: Any, categories: Mapping[str, Any]) -> bool:
    try:
        return column in categories
    except TypeError:  # an unhashable entry (``[]``, ``{}``) names no column
        return False


def is_missing(value: Any) -> bool:
    """A missing cell: ``None``, ``pd.NA``, ``NaT`` or a float NaN."""
    if value is None or value is pd.NA or value is pd.NaT:
        return True
    return isinstance(value, (float, np.floating)) and math.isnan(value)


def _values(series: pd.Series) -> list[Any]:
    """Non-missing row values, then the declared categories (used or not)."""
    values = [v for v in series.to_numpy(dtype=object) if not is_missing(v)]
    if isinstance(series.dtype, pd.CategoricalDtype):
        values += list(series.cat.categories)
    return values


def identity_eligible(series: pd.Series) -> bool:
    """Whether the identity key decides this column: every value is in V and
    none is a keyless temporal value (fit refuses those before asking)."""
    return not any(
        refusal(v) is not None or calendar_units(v) or holds_unitless_datetime(v)
        for v in _values(series)
    )


def cast_is_faithful(original: pd.Series, cast: pd.Series) -> bool:
    """Does pandas' cast agree with the key, row by row? (executed, not predicted)

    The categories' keys are unique, every non-missing row has a code whose
    category has the row's key, every missing row has code -1, and declared
    categories survive in their order.
    """
    cats = list(cast.cat.categories)
    if isinstance(original.dtype, pd.CategoricalDtype) and cats != list(
        original.cat.categories
    ):
        return False
    keys = [category_identity_key(c) for c in cats]
    if len(set(keys)) != len(keys):
        return False
    codes = cast.cat.codes.to_numpy()
    for raw, code in zip(original.to_numpy(dtype=object), codes, strict=True):
        if is_missing(raw):
            if code != -1:
                return False
        elif code < 0 or keys[code] != category_identity_key(raw):
            return False
    return True


def _key(value: Any) -> Any:
    if isinstance(value, _TEMPORAL):
        return temporal_key(value)
    number = _number_key(value)
    if number is not None:
        return number
    if _is(value, (str, np.str_)):
        return ("string", str(value))
    if _is(value, (bytes, np.bytes_)):
        return ("bytes", bytes(value))
    if type(value) is tuple:
        return ("tuple", tuple(_key(item) for item in value))
    if type(value) is frozenset:
        return ("frozenset", frozenset(_key(item) for item in value))
    raise AssertionError(f"unreachable: {type(value).__name__}")  # pragma: no cover
