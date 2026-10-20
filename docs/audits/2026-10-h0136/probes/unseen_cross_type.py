"""H-0136 probe (review round 1, finding 1): an unseen input value equal across types to a projected category.

A model learns ``[np.uint64(4), "x"]``. At predict an input row holds
``np.timedelta64(4, 'M')``, which the in-memory model treats as unseen. The
codegen JSON writes ``np.uint64(4)`` as ``int`` 4, which equals that input.
This probe compares, per ``unseen_policy``, the in-memory predict with the
predict of the model after ``Model.export`` + ``Model.load`` and with the
generated ``predict.py``'s transform, on the same rows. The learned values
themselves keep their codes in both (the case P1 checks).

Usage: python unseen_cross_type.py
"""

from __future__ import annotations

import importlib.util
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from lizyml import Model

rng = np.random.default_rng(0)
n = 200
values = [np.uint64(4)] * 80 + ["x"] * 120
train = pd.DataFrame(
    {
        "c": pd.Series(values, dtype=object),
        "num": rng.normal(size=n),
        "target": (pd.Series(values) == "x").astype(int) ^ (rng.random(n) < 0.1),
    }
)
probe = pd.DataFrame({"c": pd.Series([np.timedelta64(4, "M"), "x"], dtype=object), "num": [0.0, 0.0]})
print(f"pandas={pd.__version__}")
for policy in ("mode", "nan", "error"):
    cfg = {
        "config_version": 1,
        "task": "binary",
        "data": {"target": "target"},
        "features": {"categorical": ["c"], "unseen_policy": policy},
        "split": {"method": "kfold", "n_splits": 2, "random_state": 0},
        "model": {"name": "lgbm", "params": {"n_estimators": 5, "verbose": -1}},
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = Model(cfg)
        model.fit(data=train)
        out = Path(tempfile.mkdtemp()) / "code"
        model.export_code(out)
    try:
        res = model.predict(probe)
        memory = f"ok, warnings={[w[:70] for w in res.warnings]}"
    except Exception as exc:  # noqa: BLE001 -- the outcome measured
        memory = f"{type(exc).__name__}: {str(exc).splitlines()[0][:80]}"
    spec = importlib.util.spec_from_file_location(f"gen_{policy}", out / "predict.py")
    assert spec is not None and spec.loader is not None
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    try:
        generated = f"ok, codes {gen.transform(probe)['c'].tolist()}"
    except Exception as exc:  # noqa: BLE001 -- the outcome measured
        generated = f"{type(exc).__name__}: {str(exc).splitlines()[0][:80]}"
    learned = gen.transform(train.drop(columns=["target"]).head(3))["c"].tolist()
    try:
        loaded = Model.load(model.export(Path(tempfile.mkdtemp()) / "artifact")).predict(probe)
        saved = f"ok, warnings={[w[:70] for w in loaded.warnings]}"
    except Exception as exc:  # noqa: BLE001 -- the outcome measured
        saved = f"{type(exc).__name__}: {str(exc).splitlines()[0][:80]}"
    print(
        f"policy={policy}: in-memory {memory} | after Model.export+load {saved} | "
        f"generated {generated} | learned rows codes {learned}"
    )
