"""Probe: the names LightGBM keeps in a booster and in its model text (#406).

For each name: the booster's ``feature_name()`` and whether the model text
reads back. Confirms that LightGBM rewrites whitespace to ``_`` (so only the
empty name is lost from the space-separated ``feature_names=`` line).
"""

from __future__ import annotations

import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

NAMES: list[str] = ["z", "", " ", "a b", "a\tb", "a\nb", " a", "列", "a=b"]


def main() -> int:
    rng = np.random.default_rng(0)
    for name in NAMES:
        X = pd.DataFrame({"x": rng.normal(size=100), name: rng.normal(size=100)})
        y = (X["x"] > 0).astype(int)
        booster = lgb.train(
            {"objective": "binary", "verbose": -1},
            lgb.Dataset(X, y),
            num_boost_round=2,
            keep_training_booster=True,
        )
        try:
            lgb.Booster(model_str=booster.model_to_string())
            read = "reads back"
        except lgb.basic.LightGBMError as exc:
            read = f"LightGBMError: {exc}"
        print(f"{name!r}\tfeature_name()={booster.feature_name()!r}\t{read}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
