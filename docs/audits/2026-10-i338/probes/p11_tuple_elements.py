"""#338 (H-0127 review round 1): tuple category values whose elements LightGBM
cannot write, through the real writer (``Booster.model_to_string``)."""

from __future__ import annotations

import warnings
from decimal import Decimal

import lightgbm as lgb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
rng = np.random.default_rng(0)
n = 120
CASES = {
    "tuple[int]": [(1, 2), (3, 4), (5, 6)],
    "tuple[str]": [("a",), ("b",), ("c",)],
    "tuple[np.int64]": [(np.int64(1),), (np.int64(2),), (np.int64(3),)],
    "tuple[bytes]": [(b"a",), (b"b",), (b"c",)],
    "tuple[Decimal]": [(Decimal("1.5"),), (Decimal("2.5"),), (Decimal("3.5"),)],
    "tuple[np.longdouble]": [(np.longdouble(1),), (np.longdouble(2),), (np.longdouble(3),)],
    "tuple[tuple[bytes]]": [((b"a",),), ((b"b",),), ((b"c",),)],
}
for label, values in CASES.items():
    idx = rng.integers(0, 3, n)
    col = pd.Series(pd.Categorical([values[i] for i in idx], categories=values))
    X = pd.DataFrame({"d": col, "x": rng.normal(size=n)})
    y = (X["x"] > 0).astype(int)
    booster = lgb.train(
        {"objective": "binary", "verbose": -1, "min_data_in_leaf": 1},
        lgb.Dataset(X, label=y),
        num_boost_round=2,
        keep_training_booster=True,
    )
    try:
        booster.model_to_string()
        print(f"{label:<22} model_to_string: ok")
    except Exception as exc:  # noqa: BLE001 - the probe reports what escapes
        print(f"{label:<22} model_to_string: {type(exc).__name__}: {str(exc)[:60]}")
