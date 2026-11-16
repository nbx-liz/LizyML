"""Probe: names with ``=`` and the names LightGBM reads back (#406, H-0147).

The acceptance-5 test found that ``"=b"`` as the first feature reads back
with other names. For each name and position, print the booster's names and
the read-back booster's names (or the error), and whether predictions match.
"""

from __future__ import annotations

import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

NAMES: list[str] = ["=b", "==b", "a=b", "=", "a=", "b=c=d", "x=1"]


def main() -> int:
    rng = np.random.default_rng(0)
    for name in NAMES:
        for where, columns in (("first", [name, "x"]), ("last", ["x", name])):
            X = pd.DataFrame({c: rng.normal(size=200) for c in columns})
            y = (X[columns[0]] + X[columns[1]] > 0).astype(int)
            booster = lgb.train(
                {"objective": "binary", "verbose": -1},
                lgb.Dataset(X, y),
                num_boost_round=5,
                keep_training_booster=True,
            )
            try:
                back = lgb.Booster(model_str=booster.model_to_string())
            except lgb.basic.LightGBMError as exc:
                print(f"{name!r}\t{where}\tLightGBMError: {exc}", flush=True)
                continue
            same = np.allclose(booster.predict(X.to_numpy()), back.predict(X.to_numpy()))
            print(
                f"{name!r}\t{where}\t{booster.feature_name()!r} -> {back.feature_name()!r}"
                f"\tsame predictions={same}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
