"""Probe: duplicate column names on the Config role columns (#409, H-0148 review round 1).

Each case puts a duplicate label on one role column (target, time_col,
group_col, the blocked split's columns) and prints what fit does today.
"""

from __future__ import annotations

import sys
import warnings
from typing import Any

import numpy as np
import pandas as pd

from lizyml import Model
from lizyml.core.exceptions import LizyMLError

N = 120


def base(split: dict[str, Any], data: dict[str, Any], **extra: Any) -> dict[str, Any]:
    return {
        "config_version": 1,
        "task": "binary",
        "data": {"target": "target", **data},
        "split": split,
        "model": {"name": "lgbm", "params": {"n_estimators": 5}},
        **extra,
    }


def frame(columns: list[str]) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    values = {
        "x": rng.normal(size=N),
        "target": (rng.normal(size=N) > 0).astype(int),
        "g": np.repeat(np.arange(12), N // 12),
        "t": np.arange(N),
        "b": np.repeat(np.arange(6), N // 6),
    }
    return pd.DataFrame({i: values[c] for i, c in enumerate(columns)}).set_axis(
        columns, axis=1
    )


KFOLD = {"method": "kfold", "n_splits": 3}
CASES: dict[str, tuple[list[str], dict[str, Any]]] = {
    "target twice, excluded": (
        ["x", "target", "target"],
        base(KFOLD, {}, features={"exclude": ["target"]}),
    ),
    "target twice": (["x", "target", "target"], base(KFOLD, {})),
    "group_col twice, kfold": (
        ["x", "g", "g", "target"],
        base(KFOLD, {"group_col": "g"}),
    ),
    "group_col twice, group_kfold": (
        ["x", "g", "g", "target"],
        base({"method": "group_kfold", "n_splits": 3}, {"group_col": "g"}),
    ),
    "time_col twice, kfold": (
        ["x", "t", "t", "target"],
        base(KFOLD, {"time_col": "t"}),
    ),
    "time_col twice, time_series": (
        ["x", "t", "t", "target"],
        base({"method": "time_series", "n_splits": 3}, {"time_col": "t"}),
    ),
    "feature twice, excluded": (
        ["x", "b", "b", "target"],
        base(KFOLD, {}, features={"exclude": ["b"]}),
    ),
}


def main() -> int:
    for label, (columns, config) in CASES.items():
        df = frame(columns)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                Model(config).fit(data=df)
            result = "ok"
        except LizyMLError as exc:
            result = f"{exc.code.name} {sorted(exc.context)}"
        except Exception as exc:  # noqa: BLE001 - the probe reports every failure
            result = f"{type(exc).__name__}: {str(exc)[:70]!r}"
        print(f"{label}\tfit={result}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
