"""Probe: fit-time dtype x predict-time dtype through real Model.fit/predict.

For each fit dtype of column `a`, fit once; then predict with `a` cast to each
arrival dtype. Report: ok (and whether predictions equal the same-dtype ones),
or the exception type/code/message head.
"""

import warnings

import numpy as np
import pandas as pd

from lizyml.core.exceptions import LizyMLError
from lizyml.core.model import Model

warnings.filterwarnings("ignore")
rng = np.random.default_rng(0)
n = 200
base_a = rng.integers(0, 5, n)
b = rng.normal(size=n)
y = ((base_a + b) > 2).astype(int)


def col(kind, vals):
    vals = np.asarray(vals)
    if kind == "int64":
        return pd.Series(vals.astype("int64"))
    if kind == "float64":
        return pd.Series(vals.astype("float64"))
    if kind == "float32":
        return pd.Series(vals.astype("float32"))
    if kind == "bool":
        return pd.Series((vals % 2).astype(bool))
    if kind == "Int64":
        return pd.Series(vals.astype("int64")).astype("Int64")
    if kind == "Float64":
        return pd.Series(vals.astype("float64")).astype("Float64")
    if kind == "boolean":
        return pd.Series((vals % 2).astype(bool)).astype("boolean")
    if kind == "str":
        return pd.Series(vals.astype(str)).astype("str")
    if kind == "object_num":
        return pd.Series(list(vals.astype(int)), dtype=object)
    if kind == "category_num":
        return pd.Series(vals.astype("int64")).astype("category")
    if kind == "category_str":
        return pd.Series(vals.astype(str)).astype("category")
    if kind == "datetime":
        return pd.Series(pd.to_datetime(vals.astype("int64"), unit="D"))
    raise ValueError(kind)


FIT_KINDS = ["int64", "float64", "bool", "Int64", "Float64", "boolean", "category_num", "str"]
ARRIVE_KINDS = [
    "int64", "float64", "float32", "bool", "Int64", "Float64", "boolean",
    "str", "object_num", "category_num", "category_str", "datetime",
]

cfg = {
    "config_version": 1,
    "task": "binary",
    "data": {"target": "y"},
    "split": {"method": "kfold", "n_splits": 2},
    "model": {"name": "lgbm", "params": {"n_estimators": 10}},
    "evaluation": {"metrics": ["auc"]},
}

for fk in FIT_KINDS:
    df = pd.DataFrame({"a": col(fk, base_a), "b": b, "y": y})
    m = Model(cfg, data=df)
    try:
        m.fit()
    except Exception as e:  # noqa: BLE001
        print(f"FIT {fk}: FAILED {type(e).__name__}: {str(e)[:100]}")
        continue
    recorded = m.fit_result.dtypes["a"]
    ref = m.predict(df[["a", "b"]]).proba
    for ak in ARRIVE_KINDS:
        X = pd.DataFrame({"a": col(ak, base_a), "b": b})
        try:
            r = m.predict(X)
            same = np.allclose(r.proba, ref)
            out = f"ok same={same}"
        except LizyMLError as e:
            out = f"LizyMLError {e.code.value}: {e.user_message[:70]}"
        except Exception as e:  # noqa: BLE001
            out = f"{type(e).__name__}: {str(e)[:90]}"
        print(f"fit={fk:13s} recorded={recorded:9s} arrive={ak:13s} -> {out}")
