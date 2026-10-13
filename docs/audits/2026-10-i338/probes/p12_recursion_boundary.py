"""#338 (H-0127 review round 2): where a deeply nested tuple category stops
being writable, for the H-0127 predicate and for LightGBM's real writer.

The predicate serialises ``[[value]]`` (LightGBM writes a list of category
lists). LightGBM writes from ``model_to_string()`` and, in ``Model.export``,
from inside pickling, so its writer starts deeper in the interpreter stack.
Prints, per nesting depth, the predicate's answer, ``model_to_string()`` and
``pickle.dumps(booster)``.
"""

from __future__ import annotations

import json
import pickle
import sys
import warnings
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def _default(obj: Any) -> Any:
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        plain = obj.item()
        if isinstance(plain, np.generic):
            raise TypeError(type(obj).__name__)
        return plain
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(type(obj).__name__)


def predicate(value: Any) -> bool:
    try:
        json.dumps([[value]], default=_default)
    except (TypeError, ValueError, RecursionError):
        return False
    return True


def nested(depth: int, leaf: int) -> Any:
    value: Any = leaf
    for _ in range(depth):
        value = (value,)
    return value


print("recursion limit:", sys.getrecursionlimit())
rng = np.random.default_rng(0)
n = 60
for depth in range(980, 1000, 2):
    values = [nested(depth, k) for k in range(3)]
    col = pd.Series(pd.Categorical([values[i] for i in rng.integers(0, 3, n)], categories=values))
    X = pd.DataFrame({"d": col, "x": rng.normal(size=n)})
    y = (X["x"] > 0).astype(int)
    booster = lgb.train(
        {"objective": "binary", "verbose": -1, "min_data_in_leaf": 1},
        lgb.Dataset(X, label=y),
        num_boost_round=1,
        keep_training_booster=True,
    )
    results = []
    for step in (booster.model_to_string, lambda b=booster: pickle.dumps(b)):
        try:
            step()
            results.append("ok")
        except Exception as exc:  # noqa: BLE001 - the probe reports what escapes
            results.append(type(exc).__name__)
    print(f"depth {depth}: predicate={'writable' if predicate(values[0]) else 'unwritable'} model_to_string={results[0]} pickle={results[1]}")
