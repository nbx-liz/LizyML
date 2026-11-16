"""Probe: where Model.fit fails on exact duplicate column names (#409).

Prints the lizyml frames of the traceback (innermost last).
"""

from __future__ import annotations

import sys
import traceback
import warnings

import numpy as np
import pandas as pd

from lizyml import Model

CONFIG = {
    "config_version": 1,
    "task": "binary",
    "data": {"target": "target"},
    "split": {"method": "kfold", "n_splits": 3},
    "model": {"name": "lgbm", "params": {"n_estimators": 5}},
}


def main() -> int:
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(120, 3)), columns=["x", "a", "a"])
    df["target"] = (rng.normal(size=120) > 0).astype(int)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Model(CONFIG).fit(data=df)
    except Exception as exc:  # noqa: BLE001 - the probe reports the failure
        for frame in traceback.extract_tb(exc.__traceback__):
            if "/lizyml/" in frame.filename:
                print(f"{frame.filename.split('/lizyml/')[-1]}:{frame.lineno} {frame.name}")
        print(f"{type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
