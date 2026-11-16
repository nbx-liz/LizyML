"""Probe: what a booster whose feature name holds a line break reads back as (#406).

For the cases ``booster_names.py`` found to read back (a line break in the
last name, with text before it), compare the read-back booster's
``feature_name()``, feature count and predictions with the original.
"""

from __future__ import annotations

import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

NAMES: list[str] = ["a\nb", "a\rb", "a\r\nb", "a\n", "a\nobjective=regression"]


def main() -> int:
    rng = np.random.default_rng(0)
    for name in NAMES:
        X = pd.DataFrame({"x": rng.normal(size=200), name: rng.normal(size=200)})
        y = (X["x"] + X[name] > 0).astype(int)
        booster = lgb.train(
            {"objective": "binary", "verbose": -1},
            lgb.Dataset(X, y),
            num_boost_round=5,
            keep_training_booster=True,
        )
        try:
            back = lgb.Booster(model_str=booster.model_to_string())
        except lgb.basic.LightGBMError as exc:
            print(f"{name!r}\tLightGBMError: {exc}", flush=True)
            continue
        same = np.allclose(booster.predict(X.to_numpy()), back.predict(X.to_numpy()))
        print(
            f"{name!r}\tread back feature_name()={back.feature_name()!r}"
            f"\tnum_feature={back.num_feature()}\tsame predictions={same}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
