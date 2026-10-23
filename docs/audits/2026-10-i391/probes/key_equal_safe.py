"""#391 proposal review round 1: a V-safe frozenset with the same identity key.

On NumPy 1.24, ``safe = frozenset({4, timedelta(seconds=4)})`` round-trips and
stays in V while its identity key equals that of ``v = frozenset({np.uint64(4),
np.timedelta64(4, "s")})``. Reports, on the given tree:

1. a model fitted on ``[safe, "x"]`` predicting ``v`` under each unseen policy;
2. a model fitted on ``[safe, safe, v, "x"]``: fit, the refit encoder's
   categories, the export preflight, and where in the pickled fit result a
   ``numpy.timedelta64`` (only ``v`` holds one) is still found.

    PYTHONPATH=<tree> python -B docs/audits/2026-10-i391/probes/key_equal_safe.py
"""

from __future__ import annotations

import datetime as dt
import pickle
import warnings
from typing import Any

import numpy as np
import pandas as pd

import lizyml
from lizyml import Model
from lizyml.core.category_values import category_identity_key, refusal
from lizyml.persistence.exporter import check_category_values
from tests._helpers import make_config

warnings.simplefilter("ignore")
V = frozenset({np.uint64(4), np.timedelta64(4, "s")})
SAFE = frozenset({4, dt.timedelta(seconds=4)})
print("lizyml", lizyml.__file__, "numpy", np.__version__, "pandas", pd.__version__)
print("refusal(v):", refusal(V), "| refusal(safe):", refusal(SAFE))
try:
    print("same key:", category_identity_key(V) == category_identity_key(SAFE))
except TypeError as exc:
    print("key of v:", type(exc).__name__, exc)


def fit(values: list[Any], policy: str = "mode") -> tuple[Model, pd.DataFrame]:
    rng = np.random.default_rng(0)
    rows = values * 20
    df = pd.DataFrame(
        {
            "c": pd.Series(rows, dtype=object),
            "num": rng.normal(size=len(rows)),
            "target": rng.normal(size=len(rows)),
        }
    )
    config = make_config("regression", n_estimators=5, n_splits=2)
    config["features"] = {**config.get("features", {}), "unseen_policy": policy}
    model = Model(config)
    model.fit(data=df)
    return model, df


for policy in ("mode", "nan", "error"):
    model, df = fit([SAFE, "x"], policy)
    probe = df.drop(columns=["target"]).head(2).copy()
    probe["c"] = pd.Series([V, "x"], dtype=object, index=probe.index)
    try:
        out = f"ok {np.round(model.predict(probe).pred, 6).tolist()}"
    except Exception as exc:  # noqa: BLE001 -- the probe reports it
        out = f"{type(exc).__name__}: {str(exc)[:80]}"
    print(f"fit [safe, x] / predict v, {policy:5}: {out}")

try:
    model, _ = fit([SAFE, SAFE, V, "x"])
except Exception as exc:  # noqa: BLE001 -- the probe reports it
    print("fit [safe, safe, v, x]:", type(exc).__name__, str(exc)[:120])
else:
    refit = model._refit_result  # noqa: SLF001
    enc = refit.pipeline_state["encoder"]
    cats = enc["categories"]["c"]
    print("fit [safe, safe, v, x]: ok; identity_columns", enc.get("identity_columns"))
    print("  refit categories:", [type(c).__name__ + repr(sorted(map(repr, c)) if isinstance(c, frozenset) else repr(c)) for c in cats])
    try:
        check_category_values(model._fit_result, refit)  # noqa: SLF001
        print("  export preflight: passes")
    except Exception as exc:  # noqa: BLE001 -- the probe reports it
        print("  export preflight:", type(exc).__name__, str(exc)[:100])
    fit_result = model._fit_result  # noqa: SLF001
    for name in sorted(vars(fit_result)):
        blob = pickle.dumps(getattr(fit_result, name))
        if b"timedelta64" in blob:
            print(f"  numpy.timedelta64 found in fit_result.{name}")
