"""Q4/Q5: fit / predict / export / load per category value type.

For each case the categorical column is fitted, predicted, compared with an
equivalent model whose categories are str(value) in the same order, exported,
loaded and predicted again.
"""

from __future__ import annotations

import datetime as dt
import json
import sys
import tempfile
import warnings
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

from lizyml import Model
from lizyml.core.model import Model as _M  # noqa: F401
from tests._helpers import make_config

warnings.filterwarnings("ignore")

CASES = {
    "str": ["a", "b", "c"],
    "int": [1, 2, 3],
    "float": [1.5, 2.5, 3.5],
    "bool": [True, False],
    "tuple": [(1, 2), (3, 4), (5, 6)],
    "frozenset": [frozenset({1}), frozenset({2}), frozenset({3})],
    "bytes": [b"a", b"b", b"c"],
    "bytearray(obj col)": None,  # unhashable; built separately
    "Decimal": [Decimal("1.5"), Decimal("2.5"), Decimal("3.5")],
    "pd.Timestamp": [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-03")],
    "datetime.date": [dt.date(2020, 1, 1), dt.date(2020, 1, 2), dt.date(2020, 1, 3)],
    "datetime.datetime": [dt.datetime(2020, 1, 1), dt.datetime(2020, 1, 2), dt.datetime(2020, 1, 3)],
    "np.datetime64": [np.datetime64("2020-01-01"), np.datetime64("2020-01-02"), np.datetime64("2020-01-03")],
    "pd.Timedelta": [pd.Timedelta(1, "D"), pd.Timedelta(2, "D"), pd.Timedelta(3, "D")],
    "complex": [1 + 1j, 2 + 2j, 3 + 3j],
    "np.int64": [np.int64(1), np.int64(2), np.int64(3)],
    "np.float32": [np.float32(1.5), np.float32(2.5), np.float32(3.5)],
    "mixed str/int/float": ["a", 1, 2.5],
    "mixed str/bytes": ["a", b"b", "c"],
    "str + None rows": ["a", "b", "c"],
    "str + NaN rows": ["a", "b", "c"],
}


def build(name, n=150, seed=0):
    rng = np.random.default_rng(seed)
    if name == "bytearray(obj col)":
        vals = [bytearray(b"a"), bytearray(b"b"), bytearray(b"c")]
        idx = rng.integers(0, 3, n)
        col = pd.Series([vals[i] for i in idx], dtype=object)
        k = 3
    else:
        vals = CASES[name]
        k = len(vals)
        idx = rng.integers(0, k, n)
        col = pd.Categorical([vals[i] for i in idx], categories=vals)
        if name.endswith("None rows") or name.endswith("NaN rows"):
            obj = pd.Series(col).astype(object)
            obj[::7] = None if "None" in name else np.nan
            col = pd.Categorical(obj, categories=vals)
    y = ((idx == 0) ^ (rng.random(n) < 0.1)).astype(int)
    return pd.DataFrame({"d": col, "target": y}), idx, k


def run(name):
    out = {"case": name}
    try:
        df, idx, k = build(name)
    except Exception as exc:  # noqa: BLE001
        out["build"] = f"{type(exc).__name__}: {exc}"[:90]
        return out
    out["cat_dtype"] = str(df["d"].dtype)[:40]
    try:
        m = Model(make_config("binary", n_estimators=5, n_splits=2))
        m.fit(data=df)
        out["fit"] = "ok"
    except Exception as exc:  # noqa: BLE001
        out["fit"] = f"{type(exc).__name__}: {exc}"[:120]
        return out
    b = m._refit_result.model._model
    pc = b.pandas_categorical
    out["pandas_categorical_types"] = sorted({type(v).__name__ for c in (pc or []) for v in c})
    try:
        p = m.predict(df[["d"]]).proba
        out["predict"] = "ok"
    except Exception as exc:  # noqa: BLE001
        out["predict"] = f"{type(exc).__name__}: {exc}"[:120]
        p = None
    # Equivalent str-coded model: same codes, categories are str(value).
    if p is not None and name != "bytearray(obj col)":
        cats = list(df["d"].cat.categories)
        codes = df["d"].cat.codes.to_numpy()
        scats = [f"s{i}" for i in range(len(cats))]
        sdf = pd.DataFrame(
            {
                "d": pd.Categorical.from_codes(codes, categories=scats),
                "target": df["target"],
            }
        )
        ms = Model(make_config("binary", n_estimators=5, n_splits=2))
        ms.fit(data=sdf)
        ps = ms.predict(sdf[["d"]]).proba
        out["max|p - p_strcoded|"] = float(np.max(np.abs(np.asarray(p) - np.asarray(ps))))
    tmp = Path(tempfile.mkdtemp())
    try:
        m.export(tmp / "e")
        out["export"] = "ok"
    except Exception as exc:  # noqa: BLE001
        out["export"] = f"{type(exc).__name__}: {exc}"[-70:]
        return out
    try:
        ml = Model.load(tmp / "e")
        pl = ml.predict(df[["d"]]).proba
        out["load+predict"] = "ok"
        if p is not None:
            out["max|p - p_loaded|"] = float(np.max(np.abs(np.asarray(p) - np.asarray(pl))))
    except Exception as exc:  # noqa: BLE001
        out["load+predict"] = f"{type(exc).__name__}: {exc}"[:120]
    return out


names = sys.argv[1:] or list(CASES)
for nm in names:
    print(json.dumps(run(nm), default=str), flush=True)
