"""Probe: where None and NaN column labels merge before LightGBM (#409, H-0148 impl review round 2).

1. pandas alone: duplicated / get_loc / selecting a list of labels.
2. lgb.Dataset on a frame with None and NaN labels.
3. lizyml: the training columns after the feature pipeline.
"""

from __future__ import annotations

import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

from lizyml.features.column_check import select_training_columns


def main() -> int:
    rng = np.random.default_rng(0)
    cols = pd.Index(["x", None, np.nan], dtype=object)
    df = pd.DataFrame(rng.normal(size=(40, 3)), columns=cols)
    print(f"duplicated={df.columns.duplicated(keep=False).tolist()}")
    print(f"get_loc(None)={df.columns.get_loc(None)!r} get_loc(nan)={df.columns.get_loc(np.nan)!r}")
    picked = df[list(df.columns)]
    print(f"df[list(columns)].columns={list(picked.columns)!r} shape={picked.shape}")
    try:
        lgb.Dataset(df, (rng.normal(size=40) > 0).astype(int), params={"verbose": -1}).construct()
        print("lgb.Dataset on the frame: ok")
    except lgb.basic.LightGBMError as exc:
        print(f"lgb.Dataset on the frame: LightGBMError: {exc}")
    selected, _ = select_training_columns(df, list(df.columns))
    print(f"select_training_columns -> {list(selected.columns)!r} shape={selected.shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
