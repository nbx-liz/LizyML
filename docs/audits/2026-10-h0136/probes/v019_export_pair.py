"""H-0136 probe: does Model.export accept the uint64/timedelta64 column, and does load+predict match?

Run once with the v0.19.0 sources first on the path (``git archive v0.19.0
lizyml`` plus a ``_version.py`` stub, ``PYTHONPATH`` pointing at it) and once
with the working tree. The codegen side of the same column is
``codegen_numpy_collision.py``.

Usage: python v019_export_pair.py
"""

from __future__ import annotations

import tempfile
import warnings

import numpy as np
import pandas as pd

import lizyml
from lizyml import Model

print("lizyml", lizyml.__version__, lizyml.__file__)
pair = [np.uint64(4), np.timedelta64(4, "M")]
rng = np.random.default_rng(0)
n = 120
values = pair * 10 + ["x"] * 60 + ["y"] * 40
df = pd.DataFrame(
    {
        "c": pd.Series(values, dtype=object),
        "num": rng.normal(size=n),
        "target": (rng.random(n) < np.where(pd.Series(values) == "x", 0.2, 0.8)).astype(int),
    }
)
X = df.drop(columns=["target"])
cfg = {
    "config_version": 1,
    "task": "binary",
    "data": {"target": "target"},
    "features": {"categorical": ["c"]},
    "split": {"method": "kfold", "n_splits": 2, "random_state": 0},
    "model": {"name": "lgbm", "params": {"n_estimators": 20, "verbose": -1}},
}
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    m = Model(cfg)
    m.fit(data=df)
print("fit warnings:", [str(w.message)[:80] for w in caught if "categor" in str(w.message).lower()])
before = np.asarray(m.predict(X).proba)
d = tempfile.mkdtemp() + "/a"
try:
    m.export(d)
    print("export: ok")
    res = Model.load(d).predict(X)
    print("load+predict: max_abs_diff", float(np.max(np.abs(before - np.asarray(res.proba)))), "warnings", res.warnings)
except Exception as exc:  # noqa: BLE001
    print("export/load:", type(exc).__name__, str(exc).splitlines()[0][:150])
