"""H-0136 probe: does export_code write a project whose predict cannot run?

The third-party analysis claims that the H-0133 codec turns ``np.uint64(4)``
into ``int(4)`` but keeps ``np.timedelta64(4, 'M')``. The two projected values
are equal with equal hashes, so the generated ``predict.py`` would raise
``Categorical categories must be unique`` although ``Model.fit``,
``Model.predict`` and ``export_code`` all succeed.

For the claimed column and for a control without the timedelta64, this probe
fits, predicts, runs export_code, then imports the generated ``predict.py``
and calls its ``transform`` on the training features.

Usage: python codegen_numpy_collision.py [control]
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from lizyml import Model

CONTROL = sys.argv[1:] == ["control"]
pair = [np.uint64(4)] if CONTROL else [np.uint64(4), np.timedelta64(4, "M")]
rng = np.random.default_rng(0)
n = 120
values = pair * (20 // len(pair)) + ["x"] * 60 + ["y"] * 40
values = values + ["y"] * (n - len(values))
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


def stage(name: str, fn):  # type: ignore[no-untyped-def]
    try:
        out = fn()
        print(f"{name}: ok")
        return out
    except Exception as exc:  # noqa: BLE001 -- the outcome is the measurement
        print(f"{name}: {type(exc).__name__}: {str(exc).splitlines()[0][:110]}")
        raise SystemExit(0) from None


print(f"pandas={pd.__version__} numpy={np.__version__} control={CONTROL}")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    m = Model(cfg)
    stage("fit", lambda: m.fit(data=df))
    stage("predict", lambda: m.predict(X))
    out = Path(tempfile.mkdtemp()) / "code"
    stage("export_code", lambda: m.export_code(out))
spec = importlib.util.spec_from_file_location("generated_predict", out / "predict.py")
assert spec is not None and spec.loader is not None
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
stage("generated transform", lambda: mod.transform(X))
