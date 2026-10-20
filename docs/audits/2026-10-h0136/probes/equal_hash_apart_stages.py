"""H-0136 probe (#372): where an equal-but-hash-apart category pair fails, by hash seed.

The pair is a zoneinfo ``Timestamp`` with ``fold=1`` and the pytz ``Timestamp``
of the same instant (``==`` but ``hash`` differs, PEP 495). For each
``PYTHONHASHSEED`` the script runs one child process. The child builds a
binary-classification frame whose category column holds both values plus
plain strings in other rows, and records the outcome of each stage:

1. ``unique``: how many values ``Series.unique()`` returns for the pair (1 or 2).
2. ``fit``: ``Model.fit``.
3. ``predict``: ``predict`` on the fitted model, before export.
4. ``export``: ``Model.export``.
5. ``load_predict``: ``Model.load`` and then ``predict``.
6. ``same``: whether the loaded predictions equal the fitted ones.

A stage is ``ok``, ``skipped`` (an earlier stage failed) or the exception's
type and first line. It needs pandas 2 with pytz installed.

Usage: python equal_hash_apart_stages.py <python> <first_seed> <last_seed>
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import Counter

CHILD = r'''
import datetime as dt, json, tempfile, warnings, zoneinfo
import numpy as np, pandas as pd, pytz
from lizyml import Model

warnings.simplefilter("ignore")
ny = zoneinfo.ZoneInfo("America/New_York")
a = pd.Timestamp(dt.datetime(2024, 11, 3, 1, 30, fold=1, tzinfo=ny))
b = pd.Timestamp(pytz.timezone("America/New_York").localize(dt.datetime(2024, 11, 3, 1, 30), is_dst=False))
out = {"equal": bool(a == b), "hash_apart": hash(a) != hash(b)}
rng = np.random.default_rng(0)
n = 120
values = [a, b] * 20 + ["x", "y"] * 40
col = pd.Series(values, dtype=object)
out["unique"] = int(len(pd.Series([a, b], dtype=object).unique()))
df = pd.DataFrame({"c": col.astype("object"), "num": rng.normal(size=n), "target": rng.integers(0, 2, n)})

def stage(name, fn):
    try:
        r = fn()
        out[name] = "ok"
        return r
    except Exception as exc:
        out[name] = type(exc).__name__ + ": " + str(exc).splitlines()[0][:120]
        return None

cfg = {"config_version": 1, "task": "binary", "data": {"target": "target"},
       "features": {"categorical": ["c"]}, "split": {"method": "kfold", "n_splits": 2, "random_state": 0},
       "model": {"name": "lgbm", "params": {"n_estimators": 5, "verbose": -1}}}
m = Model(cfg)
X = df.drop(columns=["target"]).iloc[:20]
fitted = stage("fit", lambda: m.fit(data=df))
before = stage("predict", lambda: m.predict(X).pred) if out["fit"] == "ok" else None
if out["fit"] != "ok":
    out["predict"] = "skipped"
d = tempfile.mkdtemp()
if out.get("predict") == "ok":
    stage("export", lambda: m.export(d + "/a"))
else:
    out["export"] = "skipped"
if out["export"] == "ok":
    after = stage("load_predict", lambda: Model.load(d + "/a").predict(X).pred)
    out["same"] = bool(after is not None and np.array_equal(np.asarray(before), np.asarray(after)))
else:
    out["load_predict"] = "skipped"
    out["same"] = None
print("RESULT " + json.dumps(out))
'''


def main() -> int:
    python, first, last = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    rows = []
    for seed in range(first, last + 1):
        env = {**os.environ, "PYTHONHASHSEED": str(seed)}
        proc = subprocess.run([python, "-c", CHILD], env=env, capture_output=True, text=True)
        line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")), None)
        if line is None:
            print(f"seed {seed}: child failed rc={proc.returncode}: {proc.stderr.strip().splitlines()[-1:]}")
            return 1
        row = {"seed": seed, **json.loads(line[7:])}
        rows.append(row)
        print(json.dumps(row, sort_keys=True))
    keys = ("unique", "fit", "predict", "export", "load_predict", "same")
    combos = Counter(tuple(str(r[k]) for k in keys) for r in rows)
    print(f"\nseeds {first}-{last}: {len(rows)}")
    for combo, n in combos.most_common():
        print(f"{n:3d}  " + " | ".join(f"{k}={v}" for k, v in zip(keys, combo)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
