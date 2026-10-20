"""H-0136 side finding: does Model.fit fail on a column holding Decimal('1') and np.int8(1)?

The design critique of H-0136 reported that ``Decimal('1') == np.int8(1)``
raises ``TypeError`` while ``np.int8(1) == Decimal('1')`` is ``True``. It also
reported that ``Model.fit`` fails in ``CategoricalEncoder.transform`` on a
column holding both values. This probe runs the fit once per seed and prints
the outcome.

Usage: python decimal_int8_fit.py
"""

from __future__ import annotations

import os
import subprocess
import sys

CHILD = r'''
import warnings
from decimal import Decimal
import numpy as np, pandas as pd
from lizyml import Model
warnings.simplefilter("ignore")
rng = np.random.default_rng(0)
n = 120
col = pd.Series([Decimal("1"), np.int8(1), "x", "y"] * 30, dtype=object)
df = pd.DataFrame({"c": col, "num": rng.normal(size=n), "target": rng.integers(0, 2, n)})
cfg = {"config_version": 1, "task": "binary", "data": {"target": "target"},
       "features": {"categorical": ["c"]}, "split": {"method": "kfold", "n_splits": 2, "random_state": 0},
       "model": {"name": "lgbm", "params": {"n_estimators": 5, "verbose": -1}}}
try:
    Model(cfg).fit(data=df)
    print("RESULT ok")
except Exception as exc:
    print("RESULT " + type(exc).__name__ + ": " + str(exc).splitlines()[0][:120])
'''

for seed in range(5):
    env = {**os.environ, "PYTHONHASHSEED": str(seed)}
    proc = subprocess.run([sys.executable, "-c", CHILD], env=env, capture_output=True, text=True)
    line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")), proc.stderr[-200:])
    print(f"seed {seed}: {line}")
