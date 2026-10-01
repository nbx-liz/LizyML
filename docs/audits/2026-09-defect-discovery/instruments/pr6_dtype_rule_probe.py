"""Probe 2: does the proposed accept rule agree with real predict on every dtype?

Rule under test (column numeric/bool at fit):
    accept(d) = is_bool_dtype(d) or (is_numeric_dtype(d) and not is_complex_dtype(d))
Also: which dtypes can be fitted at all, what string FitResult.dtypes records
for them, and whether pandas_dtype() parses that string back.
"""

import warnings

import numpy as np
import pandas as pd
from pandas.api import types as pat

from lizyml.core.exceptions import LizyMLError
from lizyml.core.model import Model

warnings.filterwarnings("ignore")
rng = np.random.default_rng(0)
n = 200
vals = rng.integers(0, 5, n)
b = rng.normal(size=n)
y = ((vals + b) > 2).astype(int)


def make(kind):
    s = pd.Series(vals)
    table = {
        "int8": lambda: s.astype("int8"), "int16": lambda: s.astype("int16"),
        "int32": lambda: s.astype("int32"), "int64": lambda: s.astype("int64"),
        "uint8": lambda: s.astype("uint8"), "uint64": lambda: s.astype("uint64"),
        "float16": lambda: s.astype("float16"), "float32": lambda: s.astype("float32"),
        "float64": lambda: s.astype("float64"),
        "Int8": lambda: s.astype("Int8"), "Int64": lambda: s.astype("Int64"),
        "UInt32": lambda: s.astype("UInt32"), "Float32": lambda: s.astype("Float32"),
        "Float64": lambda: s.astype("Float64"),
        "Int64_na": lambda: s.astype("Int64").mask(s == 0),
        "bool": lambda: (s % 2).astype(bool), "boolean": lambda: (s % 2).astype(bool).astype("boolean"),
        "complex128": lambda: s.astype("complex128"),
        "timedelta": lambda: pd.to_timedelta(s, unit="D"),
        "datetime": lambda: pd.to_datetime(s, unit="D"),
        "datetime_tz": lambda: pd.to_datetime(s, unit="D").dt.tz_localize("UTC"),
        "period": lambda: pd.Series(pd.period_range("2000-01", periods=n, freq="M")),
        "interval": lambda: pd.Series(pd.arrays.IntervalArray.from_breaks(np.arange(n + 1))),
        "sparse_float": lambda: s.astype("float64").astype(pd.SparseDtype("float64", np.nan)),
        "str": lambda: s.astype(str).astype("str"),
        "string_python": lambda: s.astype(str).astype("string[python]"),
        "object_num": lambda: pd.Series(list(vals), dtype=object),
        "category": lambda: s.astype("category"),
        "longdouble": lambda: s.astype(np.longdouble),
    }
    try:
        import pyarrow  # noqa: F401

        table["string_pyarrow"] = lambda: s.astype(str).astype("string[pyarrow]")
        table["int64_pyarrow"] = lambda: s.astype("int64[pyarrow]")
        table["float64_pyarrow"] = lambda: s.astype("float64[pyarrow]")
        table["bool_pyarrow"] = lambda: (s % 2).astype(bool).astype("bool[pyarrow]")
    except ImportError:
        pass
    return table, (table[kind]() if kind else None)


KINDS = list(make(None)[0])


def accept_pandas(d):
    return bool(pat.is_bool_dtype(d) or (pat.is_numeric_dtype(d) and not pat.is_complex_dtype(d)))


def accept(d):
    t = getattr(d, "type", None)
    return bool(
        isinstance(t, type)
        and issubclass(t, (np.integer, np.floating, np.bool_))
        and not issubclass(t, (np.timedelta64, np.longdouble))
    )


cfg = {
    "config_version": 1, "task": "binary", "data": {"target": "y"},
    "split": {"method": "kfold", "n_splits": 2},
    "model": {"name": "lgbm", "params": {"n_estimators": 10}},
    "evaluation": {"metrics": ["auc"]},
}

print("== fit side: which dtypes fit, recorded string, parse-back, rule says numeric-at-fit")
for k in KINDS:
    df = pd.DataFrame({"a": make(k)[1], "b": b, "y": y})
    m = Model(cfg, data=df)
    try:
        m.fit()
    except Exception as e:  # noqa: BLE001
        print(f"FIT {k:15s}: FAILED {type(e).__name__}: {str(e)[:80]!r}")
        continue
    rec = m.fit_result.dtypes["a"]
    try:
        parsed = pd.api.types.pandas_dtype(rec)
        verdict = accept(parsed)
        p = f"parses -> numeric_at_fit={verdict}"
    except Exception as e:  # noqa: BLE001
        p = f"PARSE FAILS {type(e).__name__}"
    print(f"FIT {k:15s}: ok recorded={rec!r:28s} {p}")

print("== predict side: model fit on float64; arrival dtype; rule vs reality")
df = pd.DataFrame({"a": make("float64")[1], "b": b, "y": y})
m = Model(cfg, data=df)
m.fit()
disagree = 0
for k in KINDS:
    X = pd.DataFrame({"a": make(k)[1], "b": b})
    rule = accept(X["a"].dtype)
    try:
        m.predict(X)
        real = "ok"
    except LizyMLError as e:
        real = f"LizyMLError {e.code.value}"
    except Exception as e:  # noqa: BLE001
        real = f"raw {type(e).__name__}: {str(e)[:60]!r}"
    agree = (real == "ok") == rule
    disagree += not agree
    print(f"ARRIVE {k:15s} dtype={str(X['a'].dtype):28s} rule_accept={rule!s:5s} real={real}  {'AGREE' if agree else 'DISAGREE'}")
print("disagreements:", disagree, "of", len(KINDS))
