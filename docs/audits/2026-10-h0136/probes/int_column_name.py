"""H-0136 (implementation review round 1, finding 1): a categorical column named by an int.

``pipeline_state.json`` keys the categories by column name, and JSON makes
every key a string. This probe fits a model whose categorical column is named
``7`` (plain values only), runs ``export_code`` and the generated
``predict.py``'s ``predict``, and compares with ``Model.predict``. Run it with
the v0.19.0 sources first on the path to see whether the release has it.

Usage: python int_column_name.py
"""

from __future__ import annotations

import importlib.util
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import lizyml
from lizyml import Model

print(f"lizyml={lizyml.__version__} pandas={pd.__version__}")
rng = np.random.default_rng(0)
n = 300
cat = rng.choice(["a", "b", "c"], n)
df = pd.DataFrame({7: pd.Series(cat, dtype=object), "x": rng.normal(size=n)})
df["target"] = ((cat == "a") ^ (rng.random(n) < 0.1)).astype(int)
cfg = {
    "config_version": 1,
    "task": "binary",
    "data": {"target": "target"},
    # No "categorical" key: the config takes names as str, so the int-named
    # object column is picked up by auto-detection.
    "split": {"method": "kfold", "n_splits": 2, "random_state": 0},
    "model": {"name": "lgbm", "params": {"n_estimators": 10, "verbose": -1}},
}
X = df.drop(columns=["target"])
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    try:
        model = Model(cfg)
        model.fit(data=df)
        expected = np.asarray(model.predict(X).proba)
        print("fit + predict: ok")
    except Exception as exc:  # noqa: BLE001 -- the outcome measured
        print(f"fit/predict: {type(exc).__name__}: {str(exc).splitlines()[0][:120]}")
        raise SystemExit(0) from None
    out = Path(tempfile.mkdtemp()) / "code"
    try:
        model.export_code(out)
        print("export_code: ok")
    except Exception as exc:  # noqa: BLE001
        print(f"export_code: {type(exc).__name__}: {str(exc).splitlines()[0][:120]}")
        raise SystemExit(0) from None
spec = importlib.util.spec_from_file_location("gen_int_col", out / "predict.py")
assert spec is not None and spec.loader is not None
gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen)
try:
    got = np.asarray(gen.predict(X)["proba"])
    print(f"generated predict: max_abs_diff={float(np.max(np.abs(got - expected))):.3g}")
except Exception as exc:  # noqa: BLE001
    print(f"generated predict: {type(exc).__name__}: {str(exc).splitlines()[0][:120]}")
