"""templates — train.py and predict.py source code for codegen export."""

from __future__ import annotations

from lizyml.codegen.column_name_template import COLUMN_NAME_COPY
from lizyml.codegen.identity_template import IDENTITY_COPY

#: The category-value codec of ``lizyml.core.category_values`` (H-0133 P2),
#: copied into both generated scripts, which do not depend on LizyML, followed
#: by the H-0138 identity copy. A test holds this copy to the library's answers
#: over the whole value population.
_VALUE_CODEC = (
    '''\
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Category values in JSON (LizyML H-0133)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# str / int / float / bool (and numpy scalars of them) are JSON values; every
# other value is a tagged object {"$lizyml": tag, ...} with fixed fields.

import base64 as _b64
import datetime as _dt
import decimal as _decimal
import inspect as _inspect
import zoneinfo as _zoneinfo

_MAX_DEPTH = 8
_EPOCH = _dt.datetime(1970, 1, 1)
_US = _dt.timedelta(microseconds=1)
_TS_UNITS = ("s", "ms", "us", "ns")
_NP_PLAIN = (
    np.bool_, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16,
    np.uint32, np.uint64, np.float32, np.float64, np.str_,
)
_NP_COMPLEX = (np.complex64, np.complex128)
_FIELDS = {
    "bytes": {"b64"}, "decimal": {"str"}, "complex": {"real", "imag"},
    "date": {"iso"}, "datetime": {"us", "fold", "tz"},
    "timestamp": {"unit", "value", "tz"},
    "datetime64": {"unit", "count", "value"},
    "timedelta64": {"unit", "count", "value"},
    "pd_timedelta": {"unit", "value"},
    "timedelta": {"days", "seconds", "microseconds"},
    "tuple": {"items"}, "frozenset": {"items"},
}


def _is(value, types):
    return any(type(value) is t for t in types)


def _tz_shape(value):
    tz = value.tzinfo
    if tz is None:
        return None
    if type(tz) is _dt.timezone:
        return {"kind": "offset", "microseconds": value.utcoffset() // _US}
    if type(tz) is _zoneinfo.ZoneInfo and tz.key is not None and (
        type(value) is pd.Timestamp or _zoneinfo.ZoneInfo(tz.key) is tz
    ):
        return {"kind": "iana", "name": tz.key}
    if type(value) is pd.Timestamp and _is_pytz_zone(tz):
        # Its own offset: restored through ZoneInfo its hash would differ.
        return {"kind": "offset", "microseconds": value.utcoffset() // _US}
    raise ValueError(f"cannot write the timezone {type(tz).__name__}")


def _is_pytz_zone(tz):
    """Whether *tz* is one of pytz's own instances for its zone name, read
    without running a descriptor of the value's class."""
    try:
        import pytz
    except ImportError:
        return False
    zone = _inspect.getattr_static(tz, "zone", None)
    if type(zone) is not str or zone not in pytz.all_timezones_set:
        return False
    base = pytz.timezone(zone)
    family = _inspect.getattr_static(base, "_tzinfos", None)
    members = family.values() if type(family) is dict else ()
    return tz is base or any(tz is m for m in members)


def _equals_itself(value):
    if _is(value, (float, np.float32, np.float64)):
        return value == value
    if type(value) is _decimal.Decimal:
        return not value.is_nan()
    if _is(value, (complex,) + _NP_COMPLEX):
        return complex(value) == complex(value)
    if _is(value, (np.datetime64, np.timedelta64)):
        return not np.isnat(value)
    return True


def _holds_frozenset(value):
    if type(value) is frozenset:
        return True
    return type(value) is tuple and any(_holds_frozenset(v) for v in value)


def _encode_value(value):
    """A category value as JSON (the same table as LizyML's export).

    A value holding a frozenset must read back as written: written and read
    with the unchecked codec, it must give back the same form (H-0145, #391).
    """
    written = _encode_raw(value)
    if _holds_frozenset(value):
        try:
            same = _encode_raw(_decode_raw(written)) == written
        except (ValueError, TypeError, OverflowError, _decimal.InvalidOperation):
            same = False
        if not same:
            raise ValueError("a frozenset whose members do not read back as written")
    return written


def _encode_raw(value, depth=0):
    """A category value as JSON, unchecked against the codec's read-back."""
    if not _equals_itself(value):
        raise ValueError(f"{value!r} equals no value, so it cannot be matched again")
    if _is(value, (str, int, float, bool)):
        return value
    if _is(value, _NP_PLAIN):
        return value.item()
    if _is(value, (bytes, np.bytes_)):
        return {"$lizyml": "bytes", "b64": _b64.b64encode(bytes(value)).decode("ascii")}
    if type(value) is _decimal.Decimal:
        return {"$lizyml": "decimal", "str": str(value)}
    if _is(value, (complex,) + _NP_COMPLEX):
        c = complex(value)
        return {"$lizyml": "complex", "real": c.real, "imag": c.imag}
    if type(value) is pd.Timestamp:
        return {"$lizyml": "timestamp", "unit": value.unit,
                "value": int(value.asm8.astype(np.int64)), "tz": _tz_shape(value)}
    if type(value) is _dt.datetime:
        wall_us = (value.replace(tzinfo=None) - _EPOCH) // _US
        return {"$lizyml": "datetime", "us": wall_us, "fold": value.fold,
                "tz": _tz_shape(value)}
    if type(value) is _dt.date:
        return {"$lizyml": "date", "iso": value.isoformat()}
    if _is(value, (np.datetime64, np.timedelta64)):
        unit, count = np.datetime_data(value.dtype)
        tag = "datetime64" if type(value) is np.datetime64 else "timedelta64"
        return {"$lizyml": tag, "unit": unit, "count": count,
                "value": int(value.astype(np.int64))}
    if type(value) is pd.Timedelta:
        return {"$lizyml": "pd_timedelta", "unit": value.unit,
                "value": int(value.asm8.astype(np.int64))}
    if type(value) is _dt.timedelta:
        return {"$lizyml": "timedelta", "days": value.days, "seconds": value.seconds,
                "microseconds": value.microseconds}
    if type(value) in (tuple, frozenset):
        if depth >= _MAX_DEPTH:
            raise ValueError(f"a tuple / frozenset nested deeper than {_MAX_DEPTH}")
        items = [_encode_raw(v, depth + 1) for v in value]
        if type(value) is frozenset:
            items.sort(key=lambda item: json.dumps(item, sort_keys=True))
            return {"$lizyml": "frozenset", "items": items}
        return {"$lizyml": "tuple", "items": items}
    raise ValueError(f"cannot write a category value of type {type(value).__name__}")


def _field(obj, name, kind):
    value = obj[name]
    if type(value) is not kind:
        raise ValueError(f"field {name!r} is not a {kind.__name__}")
    return value


def _tz(shape):
    if shape is None:
        return None
    keys = set(shape) if type(shape) is dict else set()
    if keys == {"kind", "name"} and shape["kind"] == "iana":
        return _zoneinfo.ZoneInfo(_field(shape, "name", str))
    if keys == {"kind", "microseconds"} and shape["kind"] == "offset":
        offset = _dt.timedelta(microseconds=_field(shape, "microseconds", int))
        return _dt.timezone(offset)
    raise ValueError("the timezone has an unknown shape")


def _decode_raw(obj, depth=0):
    if _is(obj, (str, int, float, bool)):
        return obj
    if type(obj) is not dict or "$lizyml" not in obj:
        raise ValueError(f"a {type(obj).__name__} is not a category value")
    tag = obj["$lizyml"]
    if type(tag) is not str or tag not in _FIELDS:
        raise ValueError(f"unknown tag {tag!r}")
    if set(obj) - {"$lizyml"} != _FIELDS[tag]:
        raise ValueError(f"tag {tag!r} needs exactly the fields {sorted(_FIELDS[tag])}")
    if tag == "bytes":
        return _b64.b64decode(_field(obj, "b64", str).encode("ascii"), validate=True)
    if tag == "decimal":
        return _decimal.Decimal(_field(obj, "str", str))
    if tag == "complex":
        return complex(_field(obj, "real", float), _field(obj, "imag", float))
    if tag == "date":
        return _dt.date.fromisoformat(_field(obj, "iso", str))
    if tag == "datetime":
        fold = _field(obj, "fold", int)
        if fold not in (0, 1):
            raise ValueError("fold is not 0 or 1")
        wall = _EPOCH + _dt.timedelta(microseconds=_field(obj, "us", int))
        return wall.replace(tzinfo=_tz(obj["tz"]), fold=fold)
    if tag in ("timestamp", "pd_timedelta"):
        unit = _field(obj, "unit", str)
        if unit not in _TS_UNITS:
            raise ValueError(f"unit {unit!r} is not one of {_TS_UNITS}")
        if tag == "pd_timedelta":
            return pd.Timedelta(np.timedelta64(_field(obj, "value", int), unit))
        stamp = pd.Timestamp(np.datetime64(_field(obj, "value", int), unit))
        tz = _tz(obj["tz"])
        return stamp if tz is None else stamp.tz_localize("UTC").tz_convert(tz)
    if tag in ("datetime64", "timedelta64"):
        unit, count = _field(obj, "unit", str), _field(obj, "count", int)
        if count < 1:
            raise ValueError("count is not positive")
        if unit == "generic" and count == 1:
            # NumPy builds no unitless datetime64 from an int (H-0139).
            return np.array(_field(obj, "value", int), dtype=np.int64).view(tag)[()]
        kind = np.datetime64 if tag == "datetime64" else np.timedelta64
        return kind(_field(obj, "value", int), unit if count == 1 else f"{count}{unit}")
    if tag == "timedelta":
        return _dt.timedelta(days=_field(obj, "days", int),
                             seconds=_field(obj, "seconds", int),
                             microseconds=_field(obj, "microseconds", int))
    items = _field(obj, "items", list)
    if depth >= _MAX_DEPTH:
        raise ValueError(f"nested deeper than {_MAX_DEPTH}")
    decoded = [_decode_raw(item, depth + 1) for item in items]
    return tuple(decoded) if tag == "tuple" else frozenset(decoded)


def _decode_value(obj):
    """The value _encode_value (or LizyML's export) wrote as *obj*.

    A shape the encoder never writes (a NaN, NaT, a deeper nesting, a
    non-canonical form) is refused: the decoded value is written again, which
    checks it, and must give back *obj* itself. Members that cannot be
    compared while the set is rebuilt are refused the same way (H-0145).
    """
    try:
        value = _decode_raw(obj)
    except (TypeError, OverflowError) as exc:
        raise ValueError(f"its members cannot be read back: {exc}") from exc
    if _encode_value(value) != obj:
        raise ValueError("it is not the form the encoder writes for its value")
    return value

'''
    + IDENTITY_COPY
    + COLUMN_NAME_COPY
)

_TRAIN_PY = '''\
#!/usr/bin/env python3
"""Train a LightGBM model and fit a probability calibrator.

Usage:
    python train.py train_data.csv
    python train.py train_data.parquet --no-calibration

Steps:
    1. Fit feature pipeline (learn category mappings -> pipeline_state.json)
    2. Train LightGBM on full data (-> model.txt)
    3. Generate OOF scores via CV for calibration
    4. Fit calibrator on OOF scores (-> calibrator.json)

Generated by lizyml -- https://github.com/nbx-liz/LizyML
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    StratifiedGroupKFold,
    StratifiedKFold,
    StratifiedShuffleSplit,
    TimeSeriesSplit,
)
from sklearn.utils.class_weight import compute_sample_weight

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).parent
ARTIFACTS = ROOT / "artifacts"

with open(ROOT / "config.json", encoding="utf-8") as _f:
    CFG = json.load(_f)


# @@VALUE_CODEC@@


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Feature Pipeline
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def cast_categoricals(X: pd.DataFrame) -> pd.DataFrame:
    """Cast columns to ``category`` as LizyML's data builder does.

    The columns listed in ``features.categorical`` and, when
    ``auto_categorical`` is on, every string or object column, are cast with
    ``astype("category")`` before the pipeline is fitted (H-0120).

    Under the H-0138 identity, a column pandas cannot cast, or whose cast
    disagrees with the key, stays as it is: ``fit_pipeline`` learns it by key.
    A declared column is left to ``apply_declared_categories``.
    """
    rule = CFG["categorical_rule"]
    keyed = identity_version() is not None
    X = prepare_input(X).copy()  # H-0141 (idempotent on aliases)
    cast = set(rule["explicit"])
    if rule["auto"]:
        for col in X.columns:
            if pd.api.types.is_string_dtype(X[col]) or X[col].dtype == object:
                cast.add(col)
    for col in cast:
        if col not in X.columns:
            continue
        series = X[col]
        _check_keyless_temporals(_column_values(series), col)  # H-0139
        if not keyed or not _identity_eligible(_column_values(series)):
            X[col] = series.astype("category")
            continue
        if col in CFG["declared_categories"]:
            continue
        try:
            as_category = series.astype("category")
        except _CAST_ERRORS:
            continue  # pandas cannot hold this mix (H-0138)
        if _cast_is_faithful(series, as_category):
            X[col] = as_category
    return X


def identity_version():
    """The category identity of the export: ``None`` (before H-0138) or 1."""
    version = CFG["category_identity_version"]
    if version is not None:
        _check_identity_version(version, "config.json")
    return version


def _declared_values() -> dict:
    return {
        col: [_decode_value(v) for v in cats]
        for col, cats in CFG["declared_categories"].items()
    }


def fit_pipeline(df: pd.DataFrame) -> dict:
    """Learn the category codes and save the pipeline state.

    The same calls as LizyML's CategoricalEncoder.fit, on the frame
    ``cast_categoricals`` returns: a ``category`` column keeps its categories
    in order (declared, or as pandas inferred them); a string column left
    uncast takes ``sorted(unique, key=str)``. Values are kept as values, never
    as str, and a value's code is its position in the list (H-0120). The file
    holds them in LizyML's tagged JSON (H-0133); the returned state holds the
    values themselves.

    Under the H-0138 identity, a column whose values are all writable is
    learned by key (``_keyed_categories``; temporal values H-0139): the
    pandas answer where it is faithful, else categories built by key. A
    numpy timedelta64 in Y or M units is refused before learning.
    """
    expected = CFG["feature_names"]
    df = prepare_input(df)  # H-0141 (idempotent on aliases)
    missing = sorted(set(expected) - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    version = identity_version()
    declared = _declared_values() if version is not None else {}
    categories: dict[str, dict] = {}
    written: dict[str, dict] = {}
    identity_columns: list = []
    for col in CFG["categorical_features"]:
        series = df[col]
        # A declared column apply_declared_categories could not restore.
        unrestored = None if hasattr(series, "cat") else declared.get(col)
        values = _column_values(series) + (unrestored or [])
        _check_keyless_temporals(values, col)  # H-0139
        if version is not None and _identity_eligible(values):
            cats, mode = _keyed_categories(series, unrestored)
            identity_columns.append(col)
        else:
            cats, mode = _today_categories(series)
        categories[col] = {"categories": cats, "mode": mode}
        written[col] = {
            "categories": [_encode_value(v) for v in cats],
            "mode": None if mode is None else _encode_value(mode),
        }
        log.info("    %s: %d categories", col, len(cats))

    state = {
        "feature_names": expected,
        "categorical_features": CFG["categorical_features"],
        "categories": categories,
        # Keep the exported policy so predict.py applies it (H-0104).
        "unseen_policy": CFG["unseen_policy"],
    }
    if version is not None:
        state["category_identity_version"] = version
        state["identity_columns"] = identity_columns
    if check_column_name_policy(CFG):
        # H-0141: a marked project's state carries the marker too.
        state["column_name_policy"] = CFG["column_name_policy"]
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    with open(ARTIFACTS / "pipeline_state.json", "w", encoding="utf-8") as f:
        json.dump({**state, "categories": written}, f, indent=2, ensure_ascii=False)
    return state


_UNINDEXABLE_FLOATS = frozenset({np.dtype(np.float16), np.dtype(np.longdouble)}) - {
    np.dtype(np.float64)
}


def _is_unindexable_float(value: object) -> bool:
    return isinstance(value, np.floating) and value.dtype in _UNINDEXABLE_FLOATS


def comparable_values(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """float16 / longdouble values as float64, and the rows float64 changes.

    As LizyML's ``comparable_values`` (H-0127): a changed value equals no
    category and is unseen; a missing value stays missing.
    """
    unchanged = pd.Series(False, index=series.index)
    dtype = series.dtype
    if isinstance(dtype, np.dtype) and dtype in _UNINDEXABLE_FLOATS:
        values = series.to_numpy()
        as64 = values.astype(np.float64)
        changed = ~np.isnan(values) & (as64.astype(dtype) != values)
        return (
            pd.Series(as64, index=series.index, name=series.name),
            pd.Series(changed, index=series.index),
        )
    if not pd.api.types.is_object_dtype(dtype) or pd.api.types.infer_dtype(
        series, skipna=True
    ) in (
        "string", "empty",
    ):
        return series, unchanged
    # By position, not label: an index may repeat a label.
    values = series.to_numpy(dtype=object)
    positions = [i for i, v in enumerate(values) if _is_unindexable_float(v)]
    if not positions:
        return series, unchanged
    converted = values.copy()
    changed = np.zeros(len(values), dtype=bool)
    for pos in positions:
        value = values[pos]
        scalar64 = np.float64(value)
        converted[pos] = float(scalar64)
        if not np.isnan(value):
            changed[pos] = bool(scalar64.astype(value.dtype) != value)
    return (
        pd.Series(converted, index=series.index, name=series.name, dtype=object),
        pd.Series(changed, index=series.index),
    )


def encode_column(
    series: pd.Series, entry: dict, policy: str, col: str, *, keyed: bool = False
) -> np.ndarray:
    """Category codes for one column, as LizyML's CategoricalEncoder.transform.

    Unseen values follow ``unseen_policy``; the column is then cast to the
    known categories, and LightGBM receives the position in that list (a
    missing value or an unknown one is NaN). float16 / longdouble values are
    compared as float64 (H-0127).
    """
    known = entry["categories"]
    arrived = series
    series, changed = comparable_values(arrived)
    if keyed:
        # H-0138: the codes by key, never through pandas' object Index.
        codes, notes = _keyed_codes(
            series, arrived, changed, known, entry["mode"], policy, col
        )
        for note in notes:
            log.warning("%s", note)
        return codes
    if hasattr(series, "cat"):
        current = set(series.cat.categories.tolist())
    else:
        current = set(series[~changed].dropna().unique().tolist())
    unseen = current - set(known)
    if unseen or changed.any():
        shown = sorted(
            {str(v) for v in unseen} | {str(v) for v in arrived[changed]}
        )
        if policy == "error":
            raise ValueError(
                f"Column '{col}' contains unseen categories: "
                f"{shown} (unseen_policy='error')"
            )
        series = series.astype(object)
        hit = series.isin(list(unseen)) | changed
        n_rows = int(hit.sum())
        # Only values some row holds count, as in the runtime.
        present = sorted({str(v) for v in arrived[hit]})
        filler = pd.Series(
            [entry["mode"] if policy == "mode" else None] * len(series),
            index=series.index, dtype=object,
        )
        series = series.where(~hit, filler)
        if policy == "mode":
            replaced_by = f"the training mode {entry['mode']!r}"
        else:
            replaced_by = "a missing value"
        if n_rows:
            log.warning(
                "Column '%s': %d row(s) with unseen categories %s replaced by %s "
                "(unseen_policy='%s')", col, n_rows, present, replaced_by, policy,
            )
    series = series.astype("category").cat.set_categories(known)
    codes = series.cat.codes.to_numpy().astype(np.float64)
    codes[codes < 0] = np.nan
    return codes


def transform(df: pd.DataFrame, state: dict) -> pd.DataFrame:
    """Apply the fitted pipeline to a DataFrame."""
    df = prepare_input(df)  # H-0141 (idempotent on aliases)
    X = df[state["feature_names"]].copy()
    for col, entry in state["categories"].items():
        if col in X.columns:
            X[col] = encode_column(
                X[col], entry, state["unseen_policy"], col,
                keyed=col in state.get("identity_columns", ()),
            )
    return X


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Reproducing the LizyML refit (H-0120)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def check_versions() -> None:
    """Warn for each library whose version differs from the export (H-0120).

    The split, the sort and the category codes depend on these versions, so a
    different one may train a different model. Training continues.
    """
    installed = {
        "lightgbm": lgb.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
    }
    for name, recorded in CFG["_versions"].items():
        if installed.get(name) != recorded:
            message = (
                f"{name} {installed.get(name)} differs from {recorded}, the "
                "version at export; the retrained model may not match LizyML's."
            )
            warnings.warn(message, UserWarning, stacklevel=2)
            log.warning(message)


def apply_declared_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Restore the ``category`` dtype of the columns declared at fit.

    A CSV keeps neither the dtype nor the declared categories; the codes of a
    declared column are positions in its declared list, so it is restored
    before the pipeline is fitted.

    Under the H-0138 identity the codes are found by key; a declared list
    pandas cannot hold as it is leaves the column as it is, and
    ``fit_pipeline`` builds it by key in the declared order.
    """
    declared = CFG["declared_categories"]
    df = prepare_input(df)  # H-0141 (idempotent on aliases)
    if not declared:
        return df
    keyed = identity_version() is not None
    df = df.copy()
    for col, values in _declared_values().items():  # tagged JSON (H-0133)
        if col not in df.columns:
            continue
        _check_keyless_temporals(_column_values(df[col]) + list(values), col)
        if not keyed:
            df[col] = df[col].astype(pd.CategoricalDtype(categories=values))
            continue
        restored = _restore_declared(df[col], values)
        if restored is not None:
            df[col] = restored
    return df


def training_order(df: pd.DataFrame) -> np.ndarray:
    """The row order LizyML trains in: time-ordered and blocked splits sort
    every row first, with the same ``Series.argsort()`` call."""
    sp = CFG.get("split") or {}
    method = sp.get("method")
    if method in ("time_series", "purged_time_series", "group_time_series"):
        return np.asarray(df[sp["time_col"]].argsort())
    if method == "blocked_group_kfold":
        return np.asarray(df[sp["blocks"]["col"]].argsort())
    return np.arange(len(df))


def _time_holdout(n: int, ratio: float, gap: int):
    n_valid = max(1, int(n * ratio))
    if n_valid + gap >= n:
        raise ValueError(
            f"Inner validation would consume all {n} sample(s) "
            f"(n_valid={n_valid}, gap={gap}, ratio={ratio})."
        )
    idx = np.arange(n, dtype=np.intp)
    return idx[: n - n_valid - gap], idx[-n_valid:]


def _stratified_time_holdout(n: int, ratio: float, y):
    valid: list[int] = []
    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        n_valid = max(1, int(len(cls_idx) * ratio))
        valid.extend(cls_idx[-n_valid:].tolist())
    mask = np.zeros(n, dtype=bool)
    mask[np.asarray(valid, dtype=np.intp)] = True
    idx = np.arange(n, dtype=np.intp)
    if not (~mask).any():
        raise ValueError("Stratified inner validation would consume all samples.")
    return idx[~mask], idx[mask]


def _blocked_group_split(n: int, ratio: float, task: str, y, groups):
    ordered = list(dict.fromkeys(groups.tolist()))
    if len(ordered) < 4:
        log.warning("Too few groups (%d) for a group-isolated split.", len(ordered))
        if task == "regression":
            return _time_holdout(n, ratio, 0)
        return _stratified_time_holdout(n, ratio, y)
    last: dict = {}
    for i, g in enumerate(groups.tolist()):
        last[g] = i
    by_last = sorted(ordered, key=lambda g: last[g])
    if task in ("binary", "multiclass"):
        labels = {}
        for g in by_last:
            values, counts = np.unique(y[groups == g], return_counts=True)
            labels[g] = values[counts.argmax()]
        classes = np.unique(y)
        per_class: dict = {c: [] for c in classes}
        for g in by_last:
            per_class[labels[g]].append(g)
        valid_groups: set = set()
        for cls in classes:
            cls_groups = per_class[cls]
            if not cls_groups:
                continue
            n_valid = max(1, int(len(cls_groups) * ratio))
            valid_groups.update(cls_groups[-n_valid:])
    else:
        n_valid = max(1, int(len(by_last) * ratio))
        valid_groups = set(by_last[-n_valid:])
    idx = np.arange(n, dtype=np.intp)
    mask = np.isin(groups, list(valid_groups))
    return idx[~mask], idx[mask]


def inner_split(n: int, y: np.ndarray, groups, iv: dict):
    """The early-stopping split LizyML's refit made, from ``config.json``.

    Each branch mirrors one strategy of lizyml/training/inner_valid.py with the
    same numpy and scikit-learn calls and the same rounding.
    """
    method, ratio = iv["method"], iv["ratio"]
    if method == "holdout":
        if iv["stratify"]:
            sss = StratifiedShuffleSplit(
                n_splits=1, test_size=ratio, random_state=iv["random_state"],
            )
            tr, va = next(sss.split(np.arange(n), y))
            return np.sort(tr.astype(np.intp)), np.sort(va.astype(np.intp))
        rng = np.random.default_rng(iv["random_state"])
        n_valid = max(1, int(np.ceil(n * ratio)))
        if n_valid >= n:
            raise ValueError(f"Inner validation would consume all {n} sample(s).")
        perm = rng.permutation(n)
        return np.sort(perm[n_valid:]), np.sort(perm[:n_valid])
    if method == "group_holdout":
        ordered = list(dict.fromkeys(groups.tolist()))
        n_valid_groups = max(1, int(len(ordered) * ratio))
        mask = np.isin(groups, ordered[-n_valid_groups:])
        idx = np.arange(n, dtype=np.intp)
        return idx[~mask], idx[mask]
    if method == "time_holdout":
        return _time_holdout(n, ratio, iv["gap"])
    if method == "blocked_group":
        return _blocked_group_split(n, ratio, iv["task"], y, groups)
    raise ValueError(f"Unknown inner_valid method: {method!r}")


def train_lgbm(
    X: pd.DataFrame, y: pd.Series, cat_cols: list[int], groups,
) -> lgb.Booster:
    """Train LightGBM as LizyML's refit does (LGBMAdapter.fit).

    ``inner_valid`` and ``early_stopping_rounds`` are read without defaults:
    a validation set exists when ``inner_valid`` is not null, and the
    early-stopping callback only when ``early_stopping_rounds`` is not null
    as well. Row weights reach the inner-train rows only.
    """
    iv = CFG["inner_valid"]
    es_rounds = CFG["early_stopping_rounds"]
    weight = None
    if CFG["sample_weight"] == "balanced":
        weight = compute_sample_weight("balanced", y.to_numpy())
    cat = cat_cols or "auto"
    callbacks: list = [lgb.log_evaluation(period=200)]
    valid_sets = None
    valid_names = None
    if iv is not None:
        tr, va = inner_split(len(y), y.to_numpy(), groups, iv)
        train_set = lgb.Dataset(
            X.iloc[tr].reset_index(drop=True),
            label=y.iloc[tr].reset_index(drop=True),
            weight=None if weight is None else weight[tr],
            categorical_feature=cat, free_raw_data=False,
        )
        valid_set = lgb.Dataset(
            X.iloc[va].reset_index(drop=True),
            label=y.iloc[va].reset_index(drop=True),
            reference=train_set, categorical_feature=cat, free_raw_data=False,
        )
        valid_sets, valid_names = [valid_set], ["valid_0"]
        if es_rounds is not None:
            callbacks.append(
                lgb.early_stopping(stopping_rounds=es_rounds, verbose=False)
            )
        log.info("    inner valid: %d train / %d valid", len(tr), len(va))
    else:
        train_set = lgb.Dataset(
            X, label=y, weight=weight, categorical_feature=cat, free_raw_data=False,
        )

    # H-0132: the booster trains on the inner-train rows when there is an
    # inner valid set, so a training-scaled feval (MASE) is bound to those.
    fevals = build_feval_from_config(y.iloc[tr] if iv is not None else y)
    booster = lgb.train(
        CFG["lgbm_params"], train_set,
        num_boost_round=CFG["num_boost_round"],
        valid_sets=valid_sets, valid_names=valid_names,
        feval=fevals if fevals else None,
        callbacks=callbacks,
        keep_training_booster=True,
    )
    booster.save_model(str(ARTIFACTS / "model.txt"))
    log.info("    saved model.txt (best_iteration=%d)", booster.best_iteration)
    return booster


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Calibration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Custom feval metrics (H-0066)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _feval_rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2)))

def _feval_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0
    return 1.0 - ss_res / ss_tot

def _feval_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from sklearn.metrics import f1_score
    pred = (y_pred >= 0.5).astype(int) if y_pred.dtype.kind == "f" else y_pred
    average = "binary" if len(np.unique(y_true)) == 2 else "macro"
    return float(f1_score(y_true, pred, zero_division=0, average=average))

def _feval_brier(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_pred.ndim == 2:
        from sklearn.preprocessing import label_binarize
        from sklearn.metrics import brier_score_loss
        classes = np.arange(y_pred.shape[1])
        y_bin = label_binarize(y_true, classes=classes)
        per_class = [
            float(brier_score_loss(y_bin[:, k], y_pred[:, k]))
            for k in range(y_pred.shape[1])
        ]
        return float(np.mean(per_class))
    from sklearn.metrics import brier_score_loss
    return float(brier_score_loss(y_true, y_pred))

def _feval_ece(y_true: np.ndarray, y_pred: np.ndarray, *, n_bins: int = 10) -> float:
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(y_true)
    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:], strict=True):
        mask = (y_pred >= lo) & (y_pred <= hi if hi == 1.0 else y_pred < hi)
        if not mask.any():
            continue
        acc = float(np.mean(y_true[mask]))
        conf = float(np.mean(y_pred[mask]))
        ece += (mask.sum() / n) * abs(acc - conf)
    return ece

def _feval_precision_at_k(
    y_true: np.ndarray, y_pred: np.ndarray, *, k: int = 10,
) -> float:
    n = len(y_true)
    n_top = max(1, int(n * k / 100))
    top_idx = np.argsort(y_pred)[::-1][:n_top]
    return float(np.mean(y_true[top_idx]))

def _feval_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from sklearn.metrics import accuracy_score
    pred = (y_pred >= 0.5).astype(int) if y_pred.dtype.kind == "f" else y_pred
    return float(accuracy_score(y_true, pred))

def _feval_smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.abs(y_true) + np.abs(y_pred)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(denom == 0, 0.0, 2.0 * np.abs(y_true - y_pred) / denom)
    return float(np.mean(terms) * 100.0)

def _feval_wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = float(np.sum(np.abs(y_true)))
    if denom == 0.0:
        raise ValueError("WAPE is undefined when sum(|y_true|) is zero.")
    return float(np.sum(np.abs(y_true - y_pred)) / denom * 100.0)

def _train_scale_mase(y_train: np.ndarray) -> float:
    # H-0132: the mean absolute deviation of the training-side targets.
    y = np.asarray(y_train, dtype=np.float64)
    with np.errstate(invalid="ignore", over="ignore"):  # inf -> NaN, refused below
        scale = float(np.mean(np.abs(y - np.mean(y)))) if len(y) else float("nan")
    if not np.isfinite(scale) or scale == 0.0:
        raise ValueError(
            "MASE is undefined when the training targets' mean absolute "
            f"deviation is zero or not finite (got {scale})."
        )
    return scale

def _feval_mase(y_true: np.ndarray, y_pred: np.ndarray, scale: float) -> float:
    return float(np.mean(np.abs(y_true - y_pred)) / scale)


#: Metrics whose scale comes from the training-side targets (H-0132).
_TRAIN_SCALE: dict = {"mase": _train_scale_mase}

_FEVAL_REGISTRY: dict = {
    "rmsle": _feval_rmsle,
    "r2": _feval_r2,
    "f1": _feval_f1,
    "brier": _feval_brier,
    "ece": _feval_ece,
    "precision_at_k": _feval_precision_at_k,
    "accuracy": _feval_accuracy,
    "smape": _feval_smape,
    "wape": _feval_wape,
    "mase": _feval_mase,
}


def build_feval_from_config(y_train=None) -> list:
    """Build LightGBM feval callables from config.json feval_metrics.

    *y_train* is the targets of the rows the booster trains on: a metric in
    ``_TRAIN_SCALE`` (MASE) takes its scale from them, never from the scored
    rows' labels.
    """
    feval_entries = CFG.get("feval_metrics", [])
    if not feval_entries:
        return []

    task = CFG["_task"]
    fevals: list = []
    for entry in feval_entries:
        name = entry["name"]
        params = entry.get("params", {})
        greater_is_better = entry["greater_is_better"]
        needs_proba = entry["needs_proba"]
        fn = _FEVAL_REGISTRY.get(name)
        if fn is None:
            log.warning("Unknown feval metric: %s — skipping", name)
            continue
        # Build display name
        if params:
            parts = ", ".join(f"{k}={v}" for k, v in sorted(params.items()))
            display = f"{name} ({parts})"
        else:
            display = name

        call_params = params
        if name in _TRAIN_SCALE:
            if y_train is None:
                raise ValueError(
                    f"feval metric {name!r} needs the training targets: "
                    "call build_feval_from_config(y_train)."
                )
            call_params = {**params, "scale": _TRAIN_SCALE[name](y_train)}

        def _make_feval(
            _fn=fn, _name=display, _gib=greater_is_better,
            _needs_proba=needs_proba, _params=call_params, _task=task,
        ):
            def feval(y_pred, dataset):
                y_true = dataset.get_label()
                # LightGBM >= 4 passes the objective's own output: binary
                # probabilities (1-D), multiclass probabilities (2-D). Do not
                # transform them again (#306).
                proba = np.asarray(y_pred)
                if _task == "multiclass" and proba.ndim != 2:
                    raise ValueError(
                        "feval expected 2-D multiclass probabilities from "
                        f"LightGBM, got shape {proba.shape}"
                    )
                # For metrics that don't need probabilities, convert to labels
                if not _needs_proba and _task in ("binary", "multiclass"):
                    if proba.ndim == 2:
                        pred = proba.argmax(axis=1).astype(np.int64)
                    else:
                        pred = (proba >= 0.5).astype(np.int64)
                else:
                    pred = proba
                value = _fn(np.asarray(y_true), np.asarray(pred), **_params)
                return (_name, value, _gib)
            return feval

        fevals.append(_make_feval())

    return fevals


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CV split reproduction (mirrors the model's split.method; #228)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
#  The calibration OOF must respect the same time/group boundary the model
#  was trained with; a shuffled K-fold would leak across it on retrain.
#  These helpers rebuild the outer folds from config.json["split"] using the
#  same logic as LizyML (sklearn for the standard families, ported numpy for
#  purged / group-time-series / blocked-group). Rows are sorted the same way
#  (by time_col, or blocks.col) before splitting; returned indices are mapped
#  back to the input row order.

def _kfold_folds(n, y, method, shuffle, random_state, n_splits):
    cls = StratifiedKFold if method == "stratified_kfold" else KFold
    kf = cls(
        n_splits=n_splits, shuffle=shuffle,
        random_state=random_state if shuffle else None,
    )
    return list(kf.split(np.arange(n), y))


def _purged_ts_folds(n, n_splits, purge_gap, max_train, max_test):
    idx = np.arange(n)
    fold_size = n // (n_splits + 1)
    if fold_size == 0:
        raise ValueError(f"n_samples={n} too small for n_splits={n_splits}")
    folds = []
    for k in range(n_splits):
        valid_start = (k + 1) * fold_size
        valid_end = min((k + 2) * fold_size, n)
        if valid_start >= valid_end:
            continue
        train_end = (k + 1) * fold_size - purge_gap
        if train_end <= 0:
            continue
        tr = idx[:train_end]
        va = idx[valid_start:valid_end]
        if max_train is not None:
            tr = tr[-max_train:]
        if max_test is not None:
            va = va[:max_test]
        folds.append((tr, va))
    return folds


def _group_ts_folds(groups, n_splits, gap, max_train, max_test):
    groups = np.asarray(groups)
    _, first_idx = np.unique(groups, return_index=True)
    ordered_unique = groups[np.sort(first_idx)]
    n_groups = len(ordered_unique)
    if n_groups < n_splits + 1 + gap:
        raise ValueError(
            f"Not enough groups ({n_groups}) for n_splits={n_splits} gap={gap}"
        )
    gfs = n_groups // (n_splits + 1)
    folds = []
    for k in range(n_splits):
        train_group_end = (k + 1) * gfs
        valid_group_start = train_group_end + gap
        if k == n_splits - 1:
            valid_group_end = n_groups
        else:
            valid_group_end = min(valid_group_start + gfs, n_groups)
        if valid_group_start >= valid_group_end:
            continue
        tgl = ordered_unique[:train_group_end]
        vgl = ordered_unique[valid_group_start:valid_group_end]
        if max_train is not None:
            tgl = tgl[-max_train:]
        if max_test is not None:
            vgl = vgl[:max_test]
        tr = np.where(np.isin(groups, list(set(tgl))))[0]
        va = np.where(np.isin(groups, list(set(vgl))))[0]
        folds.append((tr, va))
    return folds


def _blocked_period_masks(bv, cutoffs):
    masks = [bv < cutoffs[0]]
    for i in range(1, len(cutoffs)):
        masks.append((bv >= cutoffs[i - 1]) & (bv < cutoffs[i]))
    masks.append(bv >= cutoffs[-1])
    return masks


def _blocked_split_users(all_users, y, groups, sp):
    gp = sp["groups"]
    n_splits = gp["n_splits"]
    stratify = gp["stratify"]
    shuffle = gp["shuffle"]
    rng = np.random.RandomState(sp["random_state"])
    nu = len(all_users)
    if stratify and y is not None:
        labels = np.empty(nu, dtype=y.dtype)
        for i, u in enumerate(all_users):
            uy = y[groups == u]
            vals, cnts = np.unique(uy, return_counts=True)
            labels[i] = vals[cnts.argmax()]
        per_class = []
        for cls in np.unique(labels):
            ci = np.where(labels == cls)[0]
            if shuffle:
                rng.shuffle(ci)
            per_class.append(ci)
        order = []
        max_len = max(len(c) for c in per_class)
        for i in range(max_len):
            for ci in per_class:
                if i < len(ci):
                    order.append(int(ci[i]))
        user_order = np.array(order, dtype=np.intp)
    elif shuffle:
        user_order = rng.permutation(nu)
    else:
        user_order = np.arange(nu)
    fold_sizes = np.full(n_splits, nu // n_splits, dtype=int)
    fold_sizes[: nu % n_splits] += 1
    out = []
    cur = 0
    for k in range(n_splits):
        vi = user_order[cur : cur + fold_sizes[k]]
        ti = np.concatenate([user_order[:cur], user_order[cur + fold_sizes[k] :]])
        out.append((all_users[ti], all_users[vi]))
        cur += fold_sizes[k]
    return out


def _blocked_group_folds(block_values, groups, y, sp):
    bl = sp["blocks"]
    cutoffs = sorted(bl["cutoffs"])
    mode = bl["mode"]
    train_window = bl.get("train_window")
    min_train = sp.get("min_train_rows", 1)
    min_valid = sp.get("min_valid_rows", 1)
    bv = np.asarray(block_values)
    n = len(bv)
    masks = _blocked_period_masks(bv, cutoffs)
    folds = []
    for t in range(len(cutoffs)):
        train_mask = np.zeros(n, dtype=bool)
        valid_mask = masks[t + 1]
        if mode == "expanding":
            for i in range(t + 1):
                train_mask |= masks[i]
        else:
            if train_window is None:
                raise ValueError("train_window must be set when mode='sliding'")
            for i in range(max(0, t + 1 - train_window), t + 1):
                train_mask |= masks[i]
        tpi = np.where(train_mask)[0]
        vpi = np.where(valid_mask)[0]
        if len(tpi) == 0 or len(vpi) == 0:
            continue
        all_users = np.unique(np.concatenate([groups[tpi], groups[vpi]]))
        for tu, vu in _blocked_split_users(all_users, y, groups, sp):
            tr = tpi[np.isin(groups[tpi], list(set(tu)))]
            va = vpi[np.isin(groups[vpi], list(set(vu)))]
            if len(tr) < min_train or len(va) < min_valid:
                continue
            folds.append((tr.astype(np.intp), va.astype(np.intp)))
    return folds


def _resolve_folds(df: pd.DataFrame, y: np.ndarray):
    """Return outer CV ``(train_idx, valid_idx)`` pairs in ``df`` row order,
    reproducing the model's ``split.method`` so the calibration OOF respects
    the same boundary. Falls back to shuffled K-fold for older exports that
    have no ``split`` block."""
    sp = CFG.get("split")
    n = len(y)
    seed = CFG["seed"]
    task = CFG["_task"]
    if not sp:
        # Legacy export (no split block): the original task-based shuffled
        # K-fold, preserved for back-compat.
        cls = StratifiedKFold if task == "binary" else KFold
        return list(
            cls(
                n_splits=CFG.get("calibration_n_splits", 5),
                shuffle=True, random_state=seed,
            ).split(np.arange(n), y)
        )

    method = sp["method"]
    n_splits = sp["n_splits"]
    if method in ("kfold", "stratified_kfold"):
        return _kfold_folds(
            n, y, method, sp.get("shuffle", True),
            sp.get("random_state", seed), n_splits,
        )

    if method in ("time_series", "purged_time_series", "group_time_series"):
        order = np.asarray(df[sp["time_col"]].argsort())
    elif method == "blocked_group_kfold":
        order = np.asarray(df[sp["blocks"]["col"]].argsort())
    else:  # group_kfold / stratified_group_kfold — no reordering
        order = np.arange(n)

    y_s = np.asarray(y)[order]
    groups_s = None
    if sp.get("group_col"):
        groups_s = df[sp["group_col"]].to_numpy()[order]

    if method == "time_series":
        tss = TimeSeriesSplit(
            n_splits=n_splits, gap=sp.get("gap", 0),
            max_train_size=sp.get("train_size_max"),
            test_size=sp.get("test_size_max"),
        )
        folds = list(tss.split(np.arange(n)))
    elif method == "purged_time_series":
        folds = _purged_ts_folds(
            n, n_splits, sp.get("purge_gap", 0),
            sp.get("train_size_max"), sp.get("test_size_max"),
        )
    elif method == "group_time_series":
        folds = _group_ts_folds(
            groups_s, n_splits, sp.get("gap", 0),
            sp.get("train_size_max"), sp.get("test_size_max"),
        )
    elif method == "group_kfold":
        folds = list(GroupKFold(n_splits=n_splits).split(np.arange(n), y_s, groups_s))
    elif method == "stratified_group_kfold":
        sgkf = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=sp["shuffle"],
            random_state=sp["random_state"] if sp["shuffle"] else None,
        )
        folds = list(sgkf.split(np.arange(n), y_s, groups_s))
    elif method == "blocked_group_kfold":
        bv = df[sp["blocks"]["col"]].to_numpy()[order]
        gv = df[sp["groups"]["col"]].to_numpy()[order]
        folds = _blocked_group_folds(bv, gv, y_s, sp)
    else:
        raise ValueError(f"Unsupported split.method for retrain: {method!r}")

    return [(order[np.asarray(tr)], order[np.asarray(va)]) for tr, va in folds]


def _generate_oof(X: np.ndarray, y: np.ndarray, folds) -> np.ndarray:
    """Lightweight CV to produce OOF raw scores (logits) over *folds*."""
    oof = np.full(len(y), np.nan)
    n_folds = len(folds)
    for i, (trn, val) in enumerate(folds):
        log.info("    CV fold %d/%d", i + 1, n_folds)
        ds = lgb.Dataset(X[trn], label=y[trn])
        bst = lgb.train(CFG["lgbm_params"], ds,
                        num_boost_round=CFG["num_boost_round"],
                        callbacks=[lgb.log_evaluation(0)])
        oof[val] = bst.predict(X[val], raw_score=True)
    return oof


# The calibrators are rebuilt with config.json["calibration_params"], prepared
# exactly as the fit prepared them. Each fitter mirrors its LizyML counterpart
# (lizyml/calibration/{platt,beta,isotonic}.py and _optimizer.py); the codegen
# equivalence tests pin that the two agree.

# minimize method -> (honours bounds, uses the analytic gradient)
_CAL_METHODS = {
    "L-BFGS-B": (True, True), "TNC": (True, True), "SLSQP": (True, True),
    "trust-constr": (True, True), "Powell": (True, False),
    "Nelder-Mead": (True, False), "BFGS": (False, True), "CG": (False, True),
}
# options keys scipy fills from tol, per method
_CAL_TOL_KEYS = {
    "L-BFGS-B": ("ftol", "gtol"), "TNC": ("xtol", "ftol", "gtol"),
    "SLSQP": ("ftol",), "trust-constr": ("xtol", "gtol", "barrier_tol"),
    "Powell": ("xtol", "ftol"), "Nelder-Mead": ("xatol", "fatol"),
    "BFGS": ("gtol",), "CG": ("gtol",),
}


def _is_cal_number(value) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        float(value)  # an integer too large for a float is refused, not raised
    except OverflowError:
        return False
    return True


def _check_cal_params(params: dict, n_coef: int, extra: tuple) -> None:
    """Refuse what LizyML refuses, so an edited config.json is never ignored.

    Mirrors lizyml/calibration/_optimizer.py::validate_optimizer_params. Unknown
    options keys are refused when minimize runs (see _run_minimize).
    """
    def refuse(name, detail):
        raise ValueError(f"calibration_params: {name!r} {detail}")

    accepted = {"x0", "method", "bounds", "tol", "options", *extra}
    for name in params:
        if name not in accepted:
            refuse(name, f"is not accepted; accepted: {sorted(accepted)}")
    method = params.get("method", "L-BFGS-B")
    if not isinstance(method, str) or method not in _CAL_METHODS:
        refuse("method", f"must be one of {sorted(_CAL_METHODS)}; got {method!r}")
    if "x0" in params:
        x0 = params["x0"]
        if (not isinstance(x0, list) or len(x0) != n_coef
                or not all(_is_cal_number(v) and math.isfinite(v) for v in x0)):
            refuse("x0", f"must be a list of {n_coef} finite numbers; got {x0!r}")
    if "bounds" in params:
        bounds = params["bounds"]
        shape = f"must be {n_coef} [lower, upper] lists; got {bounds!r}"
        if not _CAL_METHODS[method][0]:
            refuse("bounds", f"cannot be honoured by method {method!r}")
        if not isinstance(bounds, list) or len(bounds) != n_coef:
            refuse("bounds", shape)
        for pair in bounds:
            if not isinstance(pair, list) or len(pair) != 2 or not all(
                b is None or (_is_cal_number(b) and not math.isnan(b)) for b in pair
            ):
                refuse("bounds", shape)
            if pair[0] is not None and pair[1] is not None and pair[0] > pair[1]:
                refuse("bounds", f"has a lower bound above its upper bound: {pair!r}")
    if "tol" in params:
        tol = params["tol"]
        if not _is_cal_number(tol) or not math.isfinite(tol) or tol <= 0:
            refuse("tol", f"must be a positive finite number; got {tol!r}")
    options = params.get("options", {})
    if not isinstance(options, dict):
        refuse("options", f"must be a mapping; got {options!r}")
    smoothing = params.get("target_smoothing", True)
    if not isinstance(smoothing, bool):
        refuse("target_smoothing", f"must be true or false; got {smoothing!r}")


def _run_minimize(objective, jac, kwargs: dict, name: str):
    """Run minimize, turning scipy's unknown-option warning into a refusal.

    Only that warning is a refusal. Other OptimizeWarnings (a start outside the
    bounds, for one) are warnings the LizyML fit also lets through. A result
    that did not converge is not used (LizyML raises CALIBRATION_FAILED).
    """
    import warnings
    from scipy.optimize import OptimizeWarning, minimize
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", OptimizeWarning)
        result = minimize(objective, jac=jac, **kwargs)
    for w in caught:
        if (issubclass(w.category, OptimizeWarning)
                and str(w.message).startswith("Unknown solver options")):
            raise ValueError(f"calibration_params: 'options' {w.message}")
    for w in caught:
        warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)
    if not result.success:
        raise RuntimeError(
            f"calibration ({name}) did not converge: {result.message}. Its "
            "coefficients are not the maximum-likelihood fit; relax any "
            "optimiser limit in calibration_params (maxiter, maxfun, maxls)."
        )
    return result


def _calibration_params(config: dict) -> dict:
    """calibration_params from config.json; an export from before H-0100 has none."""
    params = config.get("calibration_params", {})
    if not isinstance(params, dict):
        raise ValueError(
            f"calibration_params: must be a mapping; got {params!r}"
        )
    return params


def _minimize_kwargs(params: dict, default_x0: list, default_options: dict) -> dict:
    """Written options beat a written tol, which beats the per-method defaults."""
    method = params.get("method", "L-BFGS-B")
    options = dict(default_options.get(method, {}))
    kwargs = {"method": method}
    if "tol" in params:
        for key in _CAL_TOL_KEYS[method]:
            options.pop(key, None)
        kwargs["tol"] = params["tol"]
    options.update(params.get("options", {}))
    kwargs["options"] = options
    kwargs["x0"] = [float(v) for v in params.get("x0", default_x0)]
    if "bounds" in params:
        kwargs["bounds"] = [tuple(pair) for pair in params["bounds"]]
    return kwargs


def _fit_platt(scores: np.ndarray, y: np.ndarray, params: dict) -> dict:
    """Platt (1999): slope and intercept by maximum likelihood, smoothed targets."""
    from scipy.special import expit
    _check_cal_params(params, 2, ("target_smoothing",))
    s = np.asarray(scores, dtype=float).ravel()
    pos = np.asarray(y, dtype=float).ravel() > 0
    n_pos = float(pos.sum())
    n_neg = float(pos.size - n_pos)
    if params.get("target_smoothing", True):
        t = np.where(pos, (n_pos + 1.0) / (n_pos + 2.0), 1.0 / (n_neg + 2.0))
    else:
        t = pos.astype(float)
    largest = float(np.max(np.abs(s))) if s.size else 0.0
    scale = largest if largest >= 30.0 else 1.0
    f = s / scale
    kwargs = _minimize_kwargs(
        params,
        [0.0, -math.log((n_neg + 1.0) / (n_pos + 1.0))],
        {"L-BFGS-B": {"gtol": 1e-6, "ftol": 64 * float(np.finfo(float).eps)}},
    )
    kwargs["x0"] = [kwargs["x0"][0] * scale, kwargs["x0"][1]]
    if "bounds" in kwargs:
        (lo, hi), b_bounds = kwargs["bounds"]
        kwargs["bounds"] = [
            (None if lo is None else lo * scale, None if hi is None else hi * scale),
            b_bounds,
        ]
    gradient = _CAL_METHODS[kwargs["method"]][1]

    def objective(c):
        z = c[0] * f + c[1]
        loss = float(np.sum(np.logaddexp(0.0, z) - t * z))
        if not gradient:
            return loss
        r = expit(z) - t
        return loss, np.array([r @ f, r.sum()])

    res = _run_minimize(objective, True if gradient else None, kwargs, "platt")
    return {"method": "platt", "a": float(res.x[0]) / scale, "b": float(res.x[1])}


def _fit_beta(scores: np.ndarray, y: np.ndarray, params: dict) -> dict:
    _check_cal_params(params, 3, ())
    s = np.clip(_sigmoid(scores), 1e-10, 1 - 1e-10)
    yf = y.astype(float)
    ls, l1s = np.log(s), np.log(1 - s)

    def nll(p):
        prob = np.clip(_sigmoid(p[0] * ls + p[1] * l1s + p[2]), 1e-10, 1 - 1e-10)
        return float(-np.sum(yf * np.log(prob) + (1 - yf) * np.log(1 - prob)))

    r = _run_minimize(
        nll, None, _minimize_kwargs(params, [1.0, 1.0, 0.0], {}), "beta"
    )
    return {"method": "beta",
            "a": float(r.x[0]), "b": float(r.x[1]), "c": float(r.x[2])}


def _fit_isotonic(scores: np.ndarray, y: np.ndarray, params: dict) -> dict:
    user = dict(params)
    num_boost_round = int(user.pop("num_boost_round", 1000))
    validation_ratio = float(user.pop("validation_ratio", 0.1))
    seed = int(user.pop("seed", CFG["seed"]))
    leaf_ratio = user.pop("min_data_in_leaf_ratio", 0.01)
    merged = {
        "objective": "binary", "metric": "binary_logloss",
        "monotone_constraints": [1], "monotone_constraints_method": "advanced",
        "num_leaves": 7, "max_depth": 3, "learning_rate": 0.03,
        "lambda_l2": 5.0, "min_gain_to_split": 0.0, "feature_fraction": 1.0,
        "bagging_fraction": 1.0, "bagging_freq": 0,
    }
    # calibration_params were checked against LightGBM's registry before the fit
    # that produced this config (H-0093), so the overlay names only LightGBM
    # parameters; the calibrator's own keys were popped above.
    merged.update(user)
    merged["monotone_constraints"] = [1]
    merged.pop("verbose", None)
    merged["verbosity"] = -1
    merged["seed"] = seed

    n = len(scores)
    if leaf_ratio is not None:
        merged["min_data_in_leaf"] = max(1, math.ceil(n * leaf_ratio))
    X_cal = scores.reshape(-1, 1)
    yf = y.astype(float)

    callbacks = [lgb.log_evaluation(period=-1)]
    if n < 20:
        ds_t = lgb.Dataset(X_cal, label=yf)
        valid_sets = []
    else:
        rng = np.random.default_rng(seed)
        n_val = max(1, int(n * validation_ratio))
        idx = rng.permutation(n)
        ds_t = lgb.Dataset(X_cal[idx[n_val:]], label=yf[idx[n_val:]])
        valid_sets = [lgb.Dataset(X_cal[idx[:n_val]], label=yf[idx[:n_val]],
                                  reference=ds_t)]
        callbacks.insert(0, lgb.early_stopping(stopping_rounds=100, verbose=False))
    bst = lgb.train(merged, ds_t, num_boost_round=num_boost_round,
                    valid_sets=valid_sets,
                    valid_names=["valid"] if valid_sets else None,
                    callbacks=callbacks)
    bst.save_model(str(ARTIFACTS / "calibrator_model.txt"))
    return {"method": "isotonic", "model_file": "calibrator_model.txt"}


_CAL_FITTERS = {"platt": _fit_platt, "beta": _fit_beta, "isotonic": _fit_isotonic}


def fit_calibrator(X: np.ndarray, y: np.ndarray, df: pd.DataFrame) -> dict | None:
    """Generate OOF scores and fit calibrator. Returns params or None."""
    method = CFG.get("calibration_method")
    if not method or CFG["_task"] != "binary":
        return None

    log.info("[3/4] Generating OOF scores ...")
    folds = _resolve_folds(df, y)
    oof = _generate_oof(X, y, folds)

    log.info("[4/4] Fitting %s calibrator ...", method)
    if method not in _CAL_FITTERS:
        raise ValueError(
            f"Unknown calibration method: {method!r}. "
            f"Supported: {list(_CAL_FITTERS)}"
        )
    # A time/group split can leave first-period / uncovered rows out of every
    # validation fold, so their OOF score stays NaN. Fit the calibrator on the
    # covered rows only (matches LizyML's cross-fit C_final; #228).
    covered = ~np.isnan(oof)
    params = _CAL_FITTERS[method](oof[covered], y[covered], _calibration_params(CFG))
    with open(ARTIFACTS / "calibrator.json", "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)
    log.info("    saved calibrator.json")
    return params


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _encode_target(y: pd.Series) -> pd.Series:
    """Map non-numeric target labels to int codes (H-0070)."""
    enc = CFG.get("target_encoder", {})
    if not enc.get("needs_encoding"):
        return y
    classes = enc["classes"]
    mapping = {c: i for i, c in enumerate(classes)}
    unseen = set(pd.Series(y).dropna().unique()) - mapping.keys()
    if unseen:
        raise ValueError(
            f"Target column contains labels not seen at fit time: "
            f"{sorted(unseen, key=str)}. Known: {classes}"
        )
    return pd.Series(y).map(mapping).astype(np.int64)


def _required_columns() -> list:
    """The columns train.py reads: features, the target and the split roles."""
    names = [*CFG["feature_names"], CFG["_target_col"]]
    sp = CFG.get("split") or {}
    for key in ("time_col", "group_col"):
        if sp.get(key):
            names.append(sp[key])
    for block in ("blocks", "groups"):
        if isinstance(sp.get(block), dict) and sp[block].get("col"):
            names.append(sp[block]["col"])
    iv = CFG.get("inner_valid") or {}
    if iv.get("group_col"):
        names.append(iv["group_col"])
    return list(dict.fromkeys(names))


def train(df: pd.DataFrame, *, calibrate: bool = True) -> None:
    check_versions()
    # H-0141: a marked project renames its input to the aliases first; the
    # markers and names are checked before the required columns are read.
    if project_uses_aliases():
        df = prepare_input(df, _required_columns())
    df = apply_declared_categories(df)
    target = CFG["_target_col"]
    y = _encode_target(df[target])
    X_raw = cast_categoricals(df[CFG["feature_names"]])

    log.info("[1/4] Fitting feature pipeline ...")
    state = fit_pipeline(X_raw)
    X = transform(X_raw, state)

    log.info("[2/4] Training LightGBM ...")
    # LightGBM reads an int entry as a position, so it gets positions (H-0142,
    # H-0141); for string names this is the same booster as names.
    categorical = set(CFG["categorical_features"])
    cat_cols = [i for i, c in enumerate(X.columns) if c in categorical]
    # The refit trains on the rows in LizyML's order; the calibration below
    # keeps the input order, as its folds are mapped back to it (H-0090).
    order = training_order(df)
    iv = CFG["inner_valid"]
    group_col = iv.get("group_col") if iv else None
    groups = df[group_col].to_numpy()[order] if group_col else None
    train_lgbm(
        X.iloc[order].reset_index(drop=True),
        y.iloc[order].reset_index(drop=True),
        cat_cols,
        groups,
    )

    if calibrate:
        fit_calibrator(X.values, y.values, df)
    else:
        log.info("[3/4] Calibration skipped")
        log.info("[4/4] --")

    log.info("Done.")


def main() -> None:
    p = argparse.ArgumentParser(description="Train LightGBM (LizyML codegen)")
    p.add_argument("data", help="CSV or Parquet file")
    p.add_argument("--no-calibration", action="store_true")
    args = p.parse_args()

    df = read_input(args.data)  # H-0141: a marked project checks the header
    if project_uses_aliases():
        # Every required alias (the target included) is named when missing.
        df = prepare_input(df, _required_columns())

    target = CFG["_target_col"]
    if target not in df.columns:
        log.error(\'Target "%s" not found. Columns: %s\', target, list(df.columns))
        sys.exit(1)

    train(df, calibrate=not args.no_calibration)


if __name__ == "__main__":
    main()
'''

_PREDICT_PY = '''\
#!/usr/bin/env python3
"""Run inference with a trained model.

Usage:
    python predict.py test_data.csv
    python predict.py test_data.csv -o predictions.csv

Generated by lizyml -- https://github.com/nbx-liz/LizyML
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).parent
ARTIFACTS = ROOT / "artifacts"

with open(ROOT / "config.json", encoding="utf-8") as _f:
    CFG = json.load(_f)


# @@VALUE_CODEC@@


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Feature Transform (predict-time: no re-fitting)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _load_pipeline() -> dict:
    path = ARTIFACTS / "pipeline_state.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run train.py first.")
    with open(path, encoding="utf-8") as f:
        state = json.load(f)
    # H-0141: the column-name marker and the marked state's names are checked
    # first, before any name or value is read.
    check_column_name_policy(CFG, state, str(path))
    # H-0138: a state without the marker keeps the pandas lookup; an unknown
    # marker is refused before any value is read.
    _, state["identity_columns"] = _identity_fields(state, str(path))
    # The categories and modes are in LizyML's tagged JSON (H-0133).
    for entry in state["categories"].values():
        entry["categories"] = [_decode_value(v) for v in entry["categories"]]
        if entry["mode"] is not None:
            entry["mode"] = _decode_value(entry["mode"])
    return state


_UNINDEXABLE_FLOATS = frozenset({np.dtype(np.float16), np.dtype(np.longdouble)}) - {
    np.dtype(np.float64)
}


def _is_unindexable_float(value: object) -> bool:
    return isinstance(value, np.floating) and value.dtype in _UNINDEXABLE_FLOATS


def comparable_values(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """float16 / longdouble values as float64, and the rows float64 changes.

    As LizyML's ``comparable_values`` (H-0127): a changed value equals no
    category and is unseen; a missing value stays missing.
    """
    unchanged = pd.Series(False, index=series.index)
    dtype = series.dtype
    if isinstance(dtype, np.dtype) and dtype in _UNINDEXABLE_FLOATS:
        values = series.to_numpy()
        as64 = values.astype(np.float64)
        changed = ~np.isnan(values) & (as64.astype(dtype) != values)
        return (
            pd.Series(as64, index=series.index, name=series.name),
            pd.Series(changed, index=series.index),
        )
    if not pd.api.types.is_object_dtype(dtype) or pd.api.types.infer_dtype(
        series, skipna=True
    ) in (
        "string", "empty",
    ):
        return series, unchanged
    # By position, not label: an index may repeat a label.
    values = series.to_numpy(dtype=object)
    positions = [i for i, v in enumerate(values) if _is_unindexable_float(v)]
    if not positions:
        return series, unchanged
    converted = values.copy()
    changed = np.zeros(len(values), dtype=bool)
    for pos in positions:
        value = values[pos]
        scalar64 = np.float64(value)
        converted[pos] = float(scalar64)
        if not np.isnan(value):
            changed[pos] = bool(scalar64.astype(value.dtype) != value)
    return (
        pd.Series(converted, index=series.index, name=series.name, dtype=object),
        pd.Series(changed, index=series.index),
    )


def encode_column(
    series: pd.Series, entry: dict, policy: str, col: str, *, keyed: bool = False
) -> np.ndarray:
    """Category codes for one column, as LizyML's CategoricalEncoder.transform.

    Unseen values follow ``unseen_policy``; the column is then cast to the
    known categories, and LightGBM receives the position in that list (a
    missing value or an unknown one is NaN). Values are compared as values,
    never as str (H-0120); float16 / longdouble values as float64 (H-0127).
    """
    known = entry["categories"]
    arrived = series
    series, changed = comparable_values(arrived)
    if keyed:
        # H-0138: the codes by key, never through pandas' object Index.
        codes, notes = _keyed_codes(
            series, arrived, changed, known, entry["mode"], policy, col
        )
        for note in notes:
            log.warning("%s", note)
        return codes
    if hasattr(series, "cat"):
        current = set(series.cat.categories.tolist())
    else:
        current = set(series[~changed].dropna().unique().tolist())
    unseen = current - set(known)
    if unseen or changed.any():
        shown = sorted(
            {str(v) for v in unseen} | {str(v) for v in arrived[changed]}
        )
        if policy == "error":
            raise ValueError(
                f"Column '{col}' contains unseen categories: "
                f"{shown} (unseen_policy='error')"
            )
        series = series.astype(object)
        hit = series.isin(list(unseen)) | changed
        n_rows = int(hit.sum())
        # Only values some row holds count, as in the runtime.
        present = sorted({str(v) for v in arrived[hit]})
        filler = pd.Series(
            [entry["mode"] if policy == "mode" else None] * len(series),
            index=series.index, dtype=object,
        )
        series = series.where(~hit, filler)
        if policy == "mode":
            replaced_by = f"the training mode {entry['mode']!r}"
        else:
            replaced_by = "a missing value"
        if n_rows:
            log.warning(
                "Column '%s': %d row(s) with unseen categories %s replaced by %s "
                "(unseen_policy='%s')", col, n_rows, present, replaced_by, policy,
            )
    series = series.astype("category").cat.set_categories(known)
    codes = series.cat.codes.to_numpy().astype(np.float64)
    codes[codes < 0] = np.nan
    return codes


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """Select expected columns and apply categorical encoding."""
    state = _load_pipeline()
    expected = state["feature_names"]
    # H-0141: a marked project renames the input to its aliases first.
    df = prepare_input(df, expected)

    missing = sorted(set(expected) - set(df.columns))
    if missing:
        raise ValueError(
            f"Missing {len(missing)} column(s): {missing}. "
            f"Expected: {expected}"
        )

    extra = sorted(set(df.columns) - set(expected))
    if extra:
        log.warning("Ignoring %d extra column(s): %s", len(extra), extra)

    X = df[expected].copy()
    policy = state["unseen_policy"]
    for col, entry in state["categories"].items():
        if col in X.columns:
            X[col] = encode_column(
                X[col], entry, policy, col, keyed=col in state["identity_columns"]
            )
    return X


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Calibration (apply only)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


def _load_calibrator() -> dict | None:
    path = ARTIFACTS / "calibrator.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def calibrate(raw_scores: np.ndarray, cal: dict) -> np.ndarray:
    """Map raw logits -> calibrated probabilities."""
    m = cal["method"]
    if m == "platt":
        return 1 / (1 + np.exp(-(cal["a"] * raw_scores + cal["b"])))
    if m == "beta":
        s = np.clip(_sigmoid(raw_scores), 1e-10, 1 - 1e-10)
        logit = cal["a"] * np.log(s) + cal["b"] * np.log(1 - s) + cal["c"]
        return np.clip(_sigmoid(logit), 0, 1)
    if m == "isotonic":
        model_file = Path(cal["model_file"]).name  # sanitize path
        bst = lgb.Booster(model_file=str(ARTIFACTS / model_file))
        return np.clip(bst.predict(raw_scores.reshape(-1, 1)), 0, 1)
    raise ValueError(
        f\'Unknown calibration method: "{m}". \'
        "Supported: platt, beta, isotonic"
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Predict
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _decode_pred(codes: np.ndarray) -> np.ndarray:
    """Map int codes back to original labels (H-0070). No-op when numeric."""
    enc = CFG.get("target_encoder", {})
    if not enc.get("needs_encoding"):
        return codes
    classes = enc["classes"]
    return np.array([classes[int(c)] for c in codes], dtype=object)


def predict(df: pd.DataFrame) -> dict[str, np.ndarray | None]:
    """Run inference. Returns {"pred": ..., "proba": ...}."""
    # Feed a numpy array (not a DataFrame) so LightGBM matches features
    # positionally by ``feature_names`` order.  Categorical columns are already
    # integer-coded by ``transform`` (matching the training category codes);
    # passing a DataFrame would trip LightGBM's pandas-categorical dtype check.
    X = transform(df).to_numpy()
    booster = lgb.Booster(model_file=str(ARTIFACTS / "model.txt"))
    task = CFG["_task"]

    if task == "regression":
        return {"pred": np.asarray(booster.predict(X), dtype=np.float64),
                "proba": None}

    if task == "binary":
        proba = np.asarray(booster.predict(X), dtype=np.float64)
        cal = _load_calibrator()
        if cal:
            logits = np.asarray(booster.predict(X, raw_score=True), dtype=np.float64)
            proba = calibrate(logits, cal)
        codes = (proba > 0.5).astype(np.int64)
        return {"pred": _decode_pred(codes), "proba": proba}

    if task == "multiclass":
        proba = np.asarray(booster.predict(X), dtype=np.float64)
        codes = np.argmax(proba, axis=1).astype(np.int64)
        return {"pred": _decode_pred(codes), "proba": proba}

    raise ValueError(f\'Unknown task: "{task}"\')


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main() -> None:
    p = argparse.ArgumentParser(description="Predict (LizyML codegen)")
    p.add_argument("data", help="CSV or Parquet file")
    p.add_argument("-o", "--output", default="predictions.csv")
    args = p.parse_args()

    df = read_input(args.data)  # H-0141: a marked project checks the header
    result = predict(df)

    out = pd.DataFrame({"pred": result["pred"]})
    if result["proba"] is not None:
        if result["proba"].ndim == 1:
            out["proba"] = result["proba"]
        else:
            for i in range(result["proba"].shape[1]):
                out[f"proba_{i}"] = result["proba"][:, i]
    out.to_csv(args.output, index=False)
    log.info("Saved %d rows -> %s", len(out), args.output)


if __name__ == "__main__":
    main()
'''

_TEST_EQUIVALENCE_PY = '''\
#!/usr/bin/env python3
"""Verify that codegen predictions match a LizyML reference.

Usage:
    python test_equivalence.py test_data.csv --reference reference_predictions.csv

Compares the output of predict.py against a reference CSV exported from
LizyML (``Model.predict()``).  The reference file must contain a ``pred``
column and optionally a ``proba`` column.

This checker imports ``predict`` and calls ``predict.predict()`` directly, so
it exercises the exact code path users run.  It never re-implements the
transform / calibration logic (a second copy would let the two paths drift
apart silently).

Generated by lizyml -- https://github.com/nbx-liz/LizyML
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import predict

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Equivalence check
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _compare(
    name: str, actual: np.ndarray, expected: np.ndarray, rtol: float
) -> bool:
    """Compare two arrays and log result."""
    if len(actual) != len(expected):
        log.error(
            "%s: FAIL — row count mismatch (codegen=%d, ref=%d)",
            name, len(actual), len(expected),
        )
        return False
    try:
        np.testing.assert_allclose(actual, expected, rtol=rtol)
        log.info("%s: PASS (rtol=%s, n=%d)", name, rtol, len(actual))
        return True
    except AssertionError as e:
        log.error("%s: FAIL\\n%s", name, e)
        return False


def check_equivalence(
    df: pd.DataFrame,
    ref: pd.DataFrame,
    *,
    rtol: float = 1e-7,
) -> bool:
    """Compare codegen predictions against reference."""
    if "pred" not in ref.columns:
        raise ValueError(
            \'"pred" column not found in reference file. \'
            f"Available columns: {list(ref.columns)}"
        )

    result = predict.predict(df)
    ok = True

    # Check pred
    if not _compare("pred", result["pred"], ref["pred"].values, rtol):
        ok = False

    # Check proba
    proba = result["proba"]
    if proba is not None:
        if proba.ndim == 1 and "proba" in ref.columns:
            # Binary: single proba column
            if not _compare("proba", proba, ref["proba"].values, rtol):
                ok = False
        elif proba.ndim == 2:
            # Multiclass: proba_0, proba_1, ...
            for i in range(proba.shape[1]):
                col = f"proba_{i}"
                if col in ref.columns:
                    if not _compare(col, proba[:, i], ref[col].values, rtol):
                        ok = False

    return ok


def main() -> None:
    p = argparse.ArgumentParser(
        description="Verify codegen predictions match LizyML reference"
    )
    p.add_argument("data", help="CSV or Parquet input data")
    p.add_argument("--reference", "-r", required=True,
                   help="Reference predictions CSV (from LizyML)")
    p.add_argument("--rtol", type=float, default=1e-7,
                   help="Relative tolerance (default: 1e-7)")
    args = p.parse_args()

    # H-0141: the same validated reader as predict.py.
    df = predict.read_input(args.data)
    ref = pd.read_csv(args.reference)

    ok = check_equivalence(df, ref, rtol=args.rtol)
    if ok:
        log.info("All checks PASSED.")
    else:
        log.error("Some checks FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
'''

#: ``pyarrow`` reads the parquet training data H-0120's promise is stated for.
#: The LightGBM line is LizyML's declared floor; a test keeps the two equal (H-0126).
_REQUIREMENTS_BASE = """\
lightgbm>=4.4
numpy
pandas
pyarrow
scikit-learn
"""


_CODEC_MARKER = "# @@VALUE_CODEC@@\n"


def _with_codec(source: str) -> str:
    assert source.count(_CODEC_MARKER) == 1  # noqa: S101 - a template invariant
    return source.replace(_CODEC_MARKER, _VALUE_CODEC)


def render_train_py() -> str:
    """Return the train.py source code."""
    return _with_codec(_TRAIN_PY)


def render_predict_py() -> str:
    """Return the predict.py source code."""
    return _with_codec(_PREDICT_PY)


def render_requirements_txt(*, uses_scipy: bool = False) -> str:
    """Return the requirements.txt content.

    ``scipy`` is listed exactly when the generated code imports it: the platt and
    beta fitters in ``train.py`` optimise with ``scipy.optimize.minimize`` (lazily,
    inside ``_fit_platt`` / ``_fit_beta``; predict-time application is pure numpy).
    Platt joined beta here in H-0100, when it moved from LogisticRegression to
    Platt's own maximum-likelihood fit. This keeps the emitted dependency set
    matching the README claim (#218).
    """
    if uses_scipy:
        return _REQUIREMENTS_BASE + "scipy\n"
    return _REQUIREMENTS_BASE


def render_test_equivalence_py() -> str:
    """Return the test_equivalence.py source code."""
    return _TEST_EQUIVALENCE_PY
