"""H-0133 design probe: can LightGBM's stored categories be position stand-ins?

Measures, on the installed LightGBM and pandas:

1. A booster trained on a category column whose categories are the raw values
   and one trained on the same column with categories renamed to their
   positions 0..k-1 give identical predictions (LightGBM trains on the codes).
2. The positions booster pickles and ``save_model``s when the raw values are
   JSON-unsafe (bytes, Decimal, Timestamp, date), while the raw booster fails.
3. The hazard of mixing forms: a raw-categories booster given a positions
   frame, and a positions booster given a raw frame. LightGBM resets the
   input's categories to the stored ones, so a mismatch turns every value
   into NaN -- a silent wrong prediction unless something refuses it.

Usage: uv run python stand_in_probe.py
"""

from __future__ import annotations

import datetime as dt
import pickle
from decimal import Decimal

import lightgbm as lgb
import numpy as np
import pandas as pd

RNG = np.random.default_rng(0)
N = 400
PARAMS = {"objective": "binary", "verbose": -1, "num_leaves": 7, "min_data_in_leaf": 5, "seed": 0}

KINDS = {
    "str": ["a", "b", "c", "d"],
    "bytes": [b"a", b"b", b"c", b"d"],
    "decimal": [Decimal("1.5"), Decimal("2.5"), Decimal("3"), Decimal("4")],
    "timestamp": [pd.Timestamp("2024-01-0%d" % i) for i in range(1, 5)],
    "date": [dt.date(2024, 1, i) for i in range(1, 5)],
}


def frame(values: list, positions: bool) -> tuple[pd.DataFrame, np.ndarray]:
    idx = RNG.integers(0, len(values), N)
    x = RNG.normal(size=N)
    y = ((idx >= 2).astype(float) + 0.3 * x + RNG.normal(scale=0.5, size=N) > 0.8).astype(int)
    col = pd.Series([values[i] for i in idx], dtype=object).astype(pd.CategoricalDtype(values))
    if positions:
        col = col.cat.rename_categories(list(range(len(values))))
    return pd.DataFrame({"c": col, "x": x}), y


def train(df: pd.DataFrame, y: np.ndarray) -> lgb.Booster:
    ds = lgb.Dataset(df, y, categorical_feature=["c"], free_raw_data=False)
    return lgb.train(PARAMS, ds, num_boost_round=20, keep_training_booster=True)


def saves(booster: lgb.Booster) -> str:
    try:
        pickle.dumps(booster)
        booster.model_to_string()
        return "ok"
    except Exception as exc:  # noqa: BLE001 -- reported
        return f"{type(exc).__name__}: {exc}"


print(f"lightgbm {lgb.__version__}, pandas {pd.__version__}")
for name, values in KINDS.items():
    state = RNG.bit_generator.state
    raw_df, y = frame(values, positions=False)
    RNG.bit_generator.state = state
    pos_df, y2 = frame(values, positions=True)
    assert (y == y2).all()
    raw_b, pos_b = train(raw_df, y), train(pos_df, y)
    diff = float(np.max(np.abs(raw_b.predict(raw_df) - pos_b.predict(pos_df))))
    print(f"[{name}] max |raw - positions| = {diff}; save raw: {saves(raw_b)}; save positions: {saves(pos_b)}")
    # Mixed forms: the stored categories decide, and a mismatch is all-NaN codes.
    mixed = pos_b.predict(raw_df) if name != "str" else None
    if mixed is not None:
        nan_pred = pos_b.predict(pos_df.assign(c=pd.Categorical([np.nan] * N, categories=list(range(4)))))
        print(
            f"[{name}] positions booster on a raw frame: equals all-missing prediction = "
            f"{bool(np.allclose(mixed, nan_pred))}, differs from correct = {bool(not np.allclose(mixed, pos_b.predict(pos_df)))}"
        )
raw_df, y = frame(KINDS["str"], positions=False)
pos_df, _ = frame(KINDS["str"], positions=True)
raw_b = train(raw_df, y)
try:
    out = raw_b.predict(pos_df)
    print(f"[str] raw booster on a positions frame: no error; differs from correct = {bool(not np.allclose(out, raw_b.predict(raw_df)))}")
except Exception as exc:  # noqa: BLE001 -- reported
    print(f"[str] raw booster on a positions frame: {type(exc).__name__}: {exc}")
