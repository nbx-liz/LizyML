"""H-0136 probe: does a pytz-only Timestamp category predict differently after load?

The third-party analysis claims that joblib changes the ``fold`` of the pytz
standard occurrence of 2024-11-03 01:30 America/New_York, and with it the hash.
A known value would then be treated as unseen after ``Model.load``. No
zoneinfo value is needed.

For each ``unseen_policy``, this probe fits a binary model on a column holding
that pytz value (not the mode) and strings. It then exports, loads and compares
the predictions with the ones before export. It also records the unseen
warnings that predict reports in ``PredictionResult.warnings`` after load.

Usage: PYTHONHASHSEED=<n> python pytz_only_silent.py [control]   (pandas 2 with pytz)
"""

from __future__ import annotations

import datetime as dt
import sys
import tempfile
import warnings

import numpy as np
import pandas as pd
import pytz

from lizyml import Model

# "control" uses an unambiguous summer time, which has a single fold; the
# mechanism predicts no change for it.
CONTROL = sys.argv[1:] == ["control"]
when = dt.datetime(2024, 6, 1, 1, 30) if CONTROL else dt.datetime(2024, 11, 3, 1, 30)
b = pd.Timestamp(pytz.timezone("America/New_York").localize(when, is_dst=False))
print(f"value={b} fold={b.fold}")
rng = np.random.default_rng(0)
n = 120
values = [b] * 20 + ["x"] * 60 + ["y"] * 40  # b is not the mode
df = pd.DataFrame(
    {
        "c": pd.Series(values, dtype=object),
        "num": rng.normal(size=n),
        "target": (rng.random(n) < np.where(pd.Series(values) == "x", 0.2, 0.8)).astype(int),
    }
)
X = df.drop(columns=["target"])

for policy in ("mode", "nan", "error"):
    cfg = {
        "config_version": 1,
        "task": "binary",
        "data": {"target": "target"},
        "features": {"categorical": ["c"], "unseen_policy": policy},
        "split": {"method": "kfold", "n_splits": 2, "random_state": 0},
        "model": {"name": "lgbm", "params": {"n_estimators": 20, "verbose": -1}},
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = Model(cfg)
        m.fit(data=df)
        before = np.asarray(m.predict(X).proba)
        d = tempfile.mkdtemp() + "/a"
        m.export(d)
    # The encoder reports unseen replacements in PredictionResult.warnings,
    # not through the warnings module.
    reported: list[str] = []
    try:
        res = Model.load(d).predict(X)
        after = np.asarray(res.proba)
        reported = list(res.warnings)
        outcome = f"max_abs_diff={float(np.max(np.abs(before - after))):.6g}"
    except Exception as exc:  # noqa: BLE001 -- the outcome is the measurement
        outcome = f"{type(exc).__name__}: {str(exc).splitlines()[0][:100]}"
    unseen = [w[:110] for w in reported if "unseen" in w.lower()]
    print(f"policy={policy:5s} {outcome} unseen_warnings={len(unseen)} {unseen[:1]}")
