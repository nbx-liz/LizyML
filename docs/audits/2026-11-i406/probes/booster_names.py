"""Probe: the names LightGBM keeps in a booster and whether its model text
reads back, with the probed name first and last (#406).

For each name and position: the booster's ``feature_name()`` and whether
``Booster(model_str=booster.model_to_string())`` succeeds.
"""

from __future__ import annotations

import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

NAMES: list[str] = [
    "z",
    "",
    " ",
    "a b",
    "a\tb",
    "\tb",
    "a\nb",
    "\nb",
    "a\n",
    "a\rb",
    "\rb",
    "a\r\nb",
    "a\x0bb",
    "a\x0cb",
    "a\x00b",
    " a",
    "列",
    "a=b",
    "=b",
    "a　b",
    "a b",
]


def probe(columns: list[str]) -> str:
    rng = np.random.default_rng(0)
    X = pd.DataFrame({c: rng.normal(size=100) for c in columns})
    y = (X[columns[0]] + X[columns[1]] > 0).astype(int)
    try:
        booster = lgb.train(
            {"objective": "binary", "verbose": -1},
            lgb.Dataset(X, y),
            num_boost_round=2,
            keep_training_booster=True,
        )
    except lgb.basic.LightGBMError as exc:
        return f"train LightGBMError: {exc}"
    try:
        lgb.Booster(model_str=booster.model_to_string())
        read = "reads back"
    except lgb.basic.LightGBMError as exc:
        read = f"LightGBMError: {exc}"
    return f"feature_name()={booster.feature_name()!r}\t{read}"


def main() -> int:
    for name in NAMES:
        for where, columns in (("last", ["x", name]), ("first", [name, "x"])):
            print(f"{name!r}\t{where}\t{probe(columns)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
