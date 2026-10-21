"""identity_template -- the H-0138 category identity, copied into generated code.

The generated ``train.py`` / ``predict.py`` do not depend on LizyML, so they
carry a copy of ``lizyml.core.category_values.category_identity_key`` and of the
keyed fit and lookup of ``lizyml.features.encoders.category_identity``. The
block follows the value codec (``templates._VALUE_CODEC``) and uses its
``_is``, ``_encode_value`` and type tables. A test holds the copy to the
library's answers.
"""

from __future__ import annotations

IDENTITY_COPY = '''\
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Category identity (LizyML H-0138)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Two values are one category exactly when their keys are equal: numbers equal
# as numbers, a string or bytes value and its NumPy twin, tuples / frozensets
# item by item. A column holding a temporal value or a value the codec cannot
# write keeps the pandas lookup.

import math as _math

_IDENTITY_VERSION = 1
_TEMPORAL = (
    _dt.date, _dt.datetime, _dt.timedelta, pd.Timestamp, pd.Timedelta,
    np.datetime64, np.timedelta64,
)
_INTEGERS = (
    bool, int, np.bool_, np.int8, np.int16, np.int32, np.int64, np.uint8,
    np.uint16, np.uint32, np.uint64,
)
_FLOATS = (float, np.float32, np.float64)
_CAST_ERRORS = (TypeError, ValueError, OverflowError)


def _check_identity_version(version, where):
    if type(version) is not int or version != _IDENTITY_VERSION:
        raise ValueError(
            f"{where} names category identity version {version!r}; this script "
            f"reads only {_IDENTITY_VERSION}"
        )
    return version


def _identity_fields(state, where):
    """``(version, identity_columns)`` of a pipeline state; ``(None, [])``
    for a state written before H-0138, which keeps the pandas lookup."""
    if "category_identity_version" not in state:
        return None, []
    version = _check_identity_version(state["category_identity_version"], where)
    columns = state.get("identity_columns")
    categories = state.get("categories")
    if (
        type(columns) is not list
        or not isinstance(categories, dict)
        or not all(_names_a_column(c, categories) for c in columns)
    ):
        raise ValueError(f"{where} has a malformed identity_columns")
    return version, list(columns)


def _names_a_column(column, categories):
    try:
        return column in categories
    except TypeError:  # an unhashable entry ([], {}) names no column
        return False


def _holds_temporal(value):
    if _is(value, (tuple, frozenset)):
        return any(_holds_temporal(item) for item in value)
    return isinstance(value, _TEMPORAL)


def _in_v(value):
    try:
        _encode_value(value)
    except ValueError:
        return False
    return True


def _is_missing(value):
    if value is None or value is pd.NA or value is pd.NaT:
        return True
    return isinstance(value, (float, np.floating)) and _math.isnan(value)


def _number_key(value):
    if _is(value, _INTEGERS):
        return ("number", "finite", int(value), 1)
    if _is(value, _FLOATS):
        x = float(value)
        if _math.isinf(x):
            return ("number", "infinity", 1 if x > 0 else -1)
        n, d = x.as_integer_ratio()
        return ("number", "finite", n, d)
    if type(value) is _decimal.Decimal:
        if value.is_infinite():
            return ("number", "infinity", -1 if value.is_signed() else 1)
        n, d = value.as_integer_ratio()
        return ("number", "finite", n, d)
    if _is(value, (complex,) + _NP_COMPLEX):
        c = complex(value)
        real = _number_key(c.real)
        if c.imag == 0:
            return real
        return ("number", "complex", real, _number_key(c.imag))
    return None


def _key(value):
    number = _number_key(value)
    if number is not None:
        return number
    if _is(value, (str, np.str_)):
        return ("string", str(value))
    if _is(value, (bytes, np.bytes_)):
        return ("bytes", bytes(value))
    if type(value) is tuple:
        return ("tuple", tuple(_key(item) for item in value))
    return ("frozenset", frozenset(_key(item) for item in value))


def _identity_key(value):
    """The structural key of *value*; a value that never runs its own ``==``."""
    if _holds_temporal(value) or not _in_v(value):
        raise TypeError(
            f"no category identity key for a {type(value).__name__} value"
        )
    return _key(value)


def _column_values(series):
    """Non-missing row values, then the declared categories (used or not)."""
    values = [v for v in series.to_numpy(dtype=object) if not _is_missing(v)]
    if isinstance(series.dtype, pd.CategoricalDtype):
        values += list(series.cat.categories)
    return values


def _identity_eligible(values):
    return not any(_holds_temporal(v) or not _in_v(v) for v in values)


def _cast_is_faithful(original, cast):
    """Does pandas' cast agree with the key, row by row?"""
    cats = list(cast.cat.categories)
    if isinstance(original.dtype, pd.CategoricalDtype) and cats != list(
        original.cat.categories
    ):
        return False
    keys = [_identity_key(c) for c in cats]
    if len(set(keys)) != len(keys):
        return False
    codes = cast.cat.codes.to_numpy()
    for raw, code in zip(original.to_numpy(dtype=object), codes):
        if _is_missing(raw):
            if code != -1:
                return False
        elif code < 0 or keys[code] != _identity_key(raw):
            return False
    return True


def _today_categories(series):
    """The categories and mode of the pandas lookup (may raise)."""
    if hasattr(series, "cat"):
        cats = list(series.cat.categories)
    else:
        cats = sorted(series.dropna().unique().tolist(), key=str)
    if not cats:
        return cats, None
    modes = series.mode()
    return cats, modes.iloc[0] if len(modes) > 0 else cats[0]


def _faithful_today(series, cats):
    keys = [_identity_key(c) for c in cats]
    if len(set(keys)) != len(keys):
        return False
    known = set(keys)
    if not all(
        _identity_key(v) in known
        for v in series.to_numpy(dtype=object)
        if not _is_missing(v)
    ):
        return False
    try:
        held = list(pd.CategoricalDtype(cats).categories)
    except _CAST_ERRORS:
        return False
    if [_identity_key(c) for c in held] != keys:
        return False
    # Executed: the training rows through the pandas lookup (may raise).
    if isinstance(series.dtype, pd.CategoricalDtype):
        coded = series
    else:
        coded = series.astype("category").cat.set_categories(cats)
    return _cast_is_faithful(series, coded)


def _rebuilt(series, declared):
    """Categories by key: the first value of each key; the mode by key counts."""
    reps = {}
    sources = list(declared) if declared is not None else []
    sources += [v for v in series.to_numpy(dtype=object) if not _is_missing(v)]
    for value in sources:
        reps.setdefault(_identity_key(value), value)
    cats = list(reps.values()) if declared is not None else sorted(
        reps.values(), key=str
    )
    if not cats:
        return cats, None
    position = {_identity_key(c): i for i, c in enumerate(cats)}
    counts = [0] * len(cats)
    for value in series.to_numpy(dtype=object):
        if not _is_missing(value):
            counts[position[_identity_key(value)]] += 1
    best = max(range(len(cats)), key=lambda i: (counts[i], -i))
    return cats, cats[best]


def _keyed_categories(series, declared=None):
    """``(categories, mode)`` of an identity column, as LizyML's fit.

    The pandas answer when it is faithful under the key, else the categories
    built by key. *declared* is the declared list of a column the script
    could not restore as a ``category`` dtype; it is built by key in that
    order.
    """
    if declared is None:
        try:
            cats, mode = _today_categories(series)
            if _faithful_today(series, cats):
                return cats, mode
        except _CAST_ERRORS:
            pass  # the pandas lookup fails on this mix: build by key
        if isinstance(series.dtype, pd.CategoricalDtype):
            declared = list(series.cat.categories)
    return _rebuilt(series, declared)


def _keyed_codes(series, arrived, changed, known, mode, policy, col):
    """Codes of *series* (``comparable_values`` of *arrived*) looked up by key.

    Returns the positions as float64 (NaN for a missing or replaced value)
    and the warnings to log; unseen values follow *policy*.
    """
    position = {_identity_key(c): i for i, c in enumerate(known)}

    def lookup(value):
        try:
            return position.get(_identity_key(value))
        except TypeError:
            return None  # a temporal value or one outside V matches nothing

    values = series.to_numpy(dtype=object)
    original = arrived.to_numpy(dtype=object)
    flags = changed.to_numpy(dtype=bool)
    codes = np.full(len(values), -1, dtype=np.int64)
    hit = np.zeros(len(values), dtype=bool)
    unseen = {}
    for i, value in enumerate(values):
        if flags[i]:
            hit[i] = True
            continue
        if _is_missing(value):
            continue
        pos = lookup(value)
        if pos is None:
            hit[i] = True
            unseen[str(value)] = None
        else:
            codes[i] = pos
    if isinstance(arrived.dtype, pd.CategoricalDtype):
        for declared in arrived.cat.categories:
            if lookup(declared) is None:
                unseen[str(declared)] = None
    notes = []
    if unseen or flags.any():
        shown = sorted(set(unseen) | {str(v) for v in original[flags]})
        if policy == "error":
            raise ValueError(
                f"Column '{col}' contains unseen categories: "
                f"{shown} (unseen_policy='error')"
            )
        n_rows = int(hit.sum())
        present = sorted({str(v) for v in original[hit]})
        if policy == "mode":
            codes[hit] = -1 if mode is None else position[_identity_key(mode)]
            replaced_by = f"the training mode {mode!r}"
        else:
            codes[hit] = -1
            replaced_by = "a missing value"
        if n_rows:
            notes.append(
                f"Column '{col}': {n_rows} row(s) with unseen categories "
                f"{present} replaced by {replaced_by} (unseen_policy='{policy}')"
            )
    out = codes.astype(np.float64)
    out[codes < 0] = np.nan
    return out, notes


def _restore_declared(series, values):
    """*series* as the declared ``category`` dtype, its codes found by key.

    ``None`` when pandas cannot hold the declared list as it is: the fit then
    builds the column by key in the declared order. A column outside the key
    is cast by pandas, as before H-0138.
    """
    dtype_values = list(values)
    if not _identity_eligible(_column_values(series) + dtype_values):
        return series.astype(pd.CategoricalDtype(categories=dtype_values))
    keys = [_identity_key(v) for v in dtype_values]
    try:
        dtype = pd.CategoricalDtype(categories=dtype_values)
    except _CAST_ERRORS:
        return None
    if [_identity_key(c) for c in dtype.categories] != keys:
        return None
    position = {}
    for i, key in enumerate(keys):
        position.setdefault(key, i)
    codes = [
        -1 if _is_missing(v) else position.get(_identity_key(v), -1)
        for v in series.to_numpy(dtype=object)
    ]
    return pd.Series(
        pd.Categorical.from_codes(codes, dtype=dtype),
        index=series.index, name=series.name,
    )

'''
