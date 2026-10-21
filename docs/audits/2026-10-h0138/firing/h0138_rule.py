"""H-0138 prototype: the structural category key and the pandas-faithful gate.

``category_key`` is the identity the decided contract describes (numbers
equal as numbers are one; strings / bytes with their NumPy twins; tuples and
frozensets item by item). ``classify`` decides, for one column, which path
the proposed mechanism takes, by running pandas' cast and checking it:

- ``temporal``: a value (recursively) is temporal -> today's pandas path;
- ``outside_v``: a value is outside H-0133's V -> today's pandas path;
- ``faithful``: the cast succeeds and agrees with the key -> today's result;
- ``cast_raised``: the cast raises -> built by key (stand-in);
- ``predicate_false``: the cast succeeds but disagrees with the key -> built
  by key (stand-in).

The last two are the conditional activation H-0138 must measure.
"""

from __future__ import annotations

import decimal
import fractions
import math
from typing import Any

import numpy as np
import pandas as pd

from lizyml.core.category_values import refusal

_TEMPORAL = (
    np.datetime64,
    np.timedelta64,
    pd.Timestamp,
    pd.Timedelta,
)


def _is_temporal(value: Any) -> bool:
    import datetime as dt

    if isinstance(value, (dt.date, dt.timedelta, *_TEMPORAL)):
        return True
    if type(value) in (tuple, frozenset):
        return any(_is_temporal(v) for v in value)
    return False


def _number(value: Any) -> Any:
    if isinstance(value, (np.datetime64, np.timedelta64)):
        return None
    if isinstance(value, (complex, np.complexfloating)):
        c = complex(value)
        real = _number(c.real)
        return real if c.imag == 0 else ("complex", real, _number(c.imag))
    if isinstance(value, (bool, int, np.bool_, np.integer)):
        f = fractions.Fraction(int(value))
        return ("finite", f.numerator, f.denominator)
    if isinstance(value, (float, np.floating)):
        x = float(value)
        if math.isinf(x):
            return ("infinity", 1 if x > 0 else -1)
        f = fractions.Fraction(x)
        return ("finite", f.numerator, f.denominator)
    if type(value) is decimal.Decimal:
        if value.is_infinite():
            return ("infinity", -1 if value.is_signed() else 1)
        f = fractions.Fraction(value)
        return ("finite", f.numerator, f.denominator)
    return None


def category_key(value: Any) -> Any:
    """A hashable key: equal keys are one category (the decided contract)."""
    n = _number(value)
    if n is not None:
        return ("number", n)
    if isinstance(value, (str, np.str_)):
        return ("string", str(value))
    if isinstance(value, (bytes, np.bytes_)):
        return ("bytes", bytes(value))
    if type(value) is tuple:
        return ("tuple", tuple(category_key(v) for v in value))
    if type(value) is frozenset:
        return ("frozenset", frozenset(category_key(v) for v in value))
    raise TypeError(f"no key for {type(value).__name__}")


def _missing(value: Any) -> bool:
    if value is None or value is pd.NA or value is pd.NaT:
        return True
    return isinstance(value, (float, np.floating)) and math.isnan(value)


def faithful(original: pd.Series, cast: pd.Series) -> bool:
    """Do pandas' categories and codes agree with the key, row by row?"""
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


def classify(series: pd.Series) -> str:
    """The path for a column of any dtype: the cast and the check are run."""
    values = [v for v in series.to_numpy(dtype=object) if not _missing(v)]
    if isinstance(series.dtype, pd.CategoricalDtype):
        values += list(series.cat.categories)  # declared, used or not
    if any(_is_temporal(v) for v in values):
        return "temporal"
    if any(refusal(v) is not None for v in values):
        return "outside_v"
    try:
        cast = series.astype("category")
    except Exception:  # noqa: BLE001 -- the outcome being classified
        return "cast_raised"
    if isinstance(series.dtype, pd.CategoricalDtype) and list(
        cast.cat.categories
    ) != list(series.cat.categories):
        return "predicate_false"  # a declaration did not survive in its order
    return "faithful" if faithful(series, cast) else "predicate_false"
