"""Probe: feature names LightGBM refuses at fit, through Model.fit and Model.tune (#409).

LightGBM 4.6.0 ``Dataset::set_feature_names`` (include/LightGBM/dataset.h)
refuses a name holding one of ``" , : [ ] { }`` (``Common::CheckAllowedJSON``)
and a name that repeats another once spaces are replaced with ``_``. Each case
prints where it failed and what the caller sees.
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
BASE = {
    "config_version": 1,
    "task": "binary",
    "data": {"target": "target"},
    "split": {"method": "kfold", "n_splits": 3},
    "model": {"name": "lgbm", "params": {"n_estimators": 5}},
}
TUNE = {"tuning": {"optuna": {"params": {"n_trials": 2}, "space": {}}}}

CASES: dict[str, list[Any]] = {
    "plain": ["x", "z"],
    'quote "': ["x", 'a"b'],
    "comma ,": ["x", "a,b"],
    "colon :": ["x", "a:b"],
    "bracket [": ["x", "a[b"],
    "bracket ]": ["x", "a]b"],
    "brace {": ["x", "a{b"],
    "brace }": ["x", "a}b"],
    "space vs underscore": ["x", "a b", "a_b"],
    "two spaces": ["x", "a  b", "a__b"],
    "int vs str": ["x", 0, "0"],
    "float vs str": ["x", 1.5, "1.5"],
    "NUL vs prefix": ["x", "a\x00b", "a"],
    "apostrophe": ["x", "a'b"],
    "backslash": ["x", "a\\b"],
    "semicolon": ["x", "a;b"],
    "exact duplicate": ["x", "a", "a"],
}


def frame(names: list[Any]) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({i: rng.normal(size=N) for i in range(len(names))})
    df.columns = pd.Index(names, dtype=object)
    df["target"] = (rng.normal(size=N) > 0).astype(int)
    return df


def outcome(call: Any) -> str:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            call()
    except LizyMLError as exc:
        return f"{exc.code.name} {sorted(exc.context)}"
    except Exception as exc:  # noqa: BLE001 - the probe reports every failure
        return f"{type(exc).__module__}.{type(exc).__name__}: {str(exc)[:70]!r}"
    return "ok"


def main() -> int:
    for label, names in CASES.items():
        df = frame(names)
        fit = outcome(lambda: Model(BASE).fit(data=df))
        tune = outcome(lambda: Model({**BASE, **TUNE}).tune(data=df))
        print(f"{label}\tfit={fit}\ttune={tune}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
