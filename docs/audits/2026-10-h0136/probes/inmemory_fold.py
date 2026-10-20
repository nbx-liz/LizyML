"""H-0136 probe: does a fitted model misjudge its own training values, with no export?

The pandas-2 census found the refit state of the H-0133 ``Timestamp pytz
fold`` case holding the pytz standard occurrence of 2024-11-03 01:30
America/New_York at ``fold=1`` right after fit, and in-memory ``predict`` on
the training rows reporting them unseen. For each case this probe fits a
binary model (the H-0133 acceptance test's frame shape: 800 rows, the
category decides the target), predicts on the training features in memory,
and prints the folds the refit state holds and the unseen rows predict
reports. Nothing is exported.

Cases: the pytz pair (DST and standard side), the pytz standard side alone
beside strings (control for the pair), and the zoneinfo pair (``fold=0`` and
``fold=1``), which needs no pytz.

Usage: python inmemory_fold.py   (pandas 2 with pytz runs every case)
"""

from __future__ import annotations

import datetime as dt
import warnings
import zoneinfo
from typing import Any

import numpy as np
import pandas as pd

import lizyml
from lizyml import Model

NY = zoneinfo.ZoneInfo("America/New_York")
CASES: dict[str, list[Any]] = {
    "zoneinfo fold 0 + fold 1": [
        pd.Timestamp(dt.datetime(2024, 11, 3, 1, 30, fold=f, tzinfo=NY)) for f in (0, 1)
    ],
}
try:
    import pytz

    _ny = pytz.timezone("America/New_York")
    _std = pd.Timestamp(_ny.localize(dt.datetime(2024, 11, 3, 1, 30), is_dst=False))
    _dst = pd.Timestamp(_ny.localize(dt.datetime(2024, 11, 3, 1, 30), is_dst=True))
    CASES["pytz DST + standard"] = [_dst, _std]
    CASES["pytz standard + str (control)"] = [_std, "x"]
except ImportError:
    pass


def _frame(values: list[Any], n: int = 800, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), n)
    idx[: len(values)] = np.arange(len(values))
    target = (idx % 2 == 0) ^ (rng.random(n) < 0.1)
    column = pd.Series([values[i] for i in idx], dtype=object)
    return pd.DataFrame({"c": column, "x": rng.normal(size=n), "target": target.astype(int)})


print(f"lizyml={lizyml.__version__} pandas={pd.__version__}")
for name, values in CASES.items():
    df = _frame(values)
    cfg = {
        "config_version": 1,
        "task": "binary",
        "data": {"target": "target"},
        "features": {"categorical": ["c"]},
        "split": {"method": "kfold", "n_splits": 2, "random_state": 42},
        "model": {"name": "lgbm", "params": {"n_estimators": 8, "verbose": -1}},
        "training": {"seed": 0},
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = Model(cfg)
        model.fit(data=df)
        result = model.predict(df.drop(columns=["target"]))
    state = model._refit_result.pipeline_state["encoder"]  # type: ignore[union-attr]
    held = [(str(v), getattr(v, "fold", None)) for v in state["categories"]["c"]]
    unseen = [w[:100] for w in result.warnings if "unseen" in w]
    print(f"{name}: input folds {[getattr(v, 'fold', None) for v in values]}")
    print(f"    refit state holds {held}")
    print(f"    in-memory predict on the training rows: {unseen or 'no unseen rows'}")
