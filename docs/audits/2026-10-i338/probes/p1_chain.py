"""Q1/Q2/Q3/Q6: call chain and per-operation behaviour for one value type."""

from __future__ import annotations

import dataclasses
import io
import pickle
import sys
import tempfile
import traceback
import warnings
from decimal import Decimal
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from lizyml import Model
from tests._helpers import make_config

warnings.filterwarnings("ignore")

KIND = sys.argv[1]
VALUES = {
    "bytes": [b"a", b"b"],
    "decimal": [Decimal("1.5"), Decimal("2.5")],
    "timestamp": [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02")],
    "str": ["a", "b"],
    "tuple": [(1, 2), (3, 4)],
}[KIND]

rng = np.random.default_rng(0)
n = 120
idx = rng.integers(0, 2, n)
d = pd.Categorical([VALUES[i] for i in idx], categories=VALUES)
y = (idx ^ (rng.random(n) < 0.1)).astype(int)
df = pd.DataFrame({"d": d, "target": y})

m = Model(make_config("binary", n_estimators=5, n_splits=2))
m.fit(data=df)
print("fit: ok")
pred = m.predict(df[["d"]])
print("predict: ok", np.round(pred.pred[:4], 4) if hasattr(pred, "pred") else "")

refit = m._refit_result
booster = refit.model._model
print("booster type:", type(booster).__name__)
print("booster.pandas_categorical:", repr(booster.pandas_categorical))
print(
    "elem types:",
    [type(v).__name__ for col in (booster.pandas_categorical or []) for v in col],
)
fold_b = m._fit_result.models[0]._model
print("fold0 booster.pandas_categorical:", repr(fold_b.pandas_categorical))
print("refit.pipeline_state type:", type(refit.pipeline_state).__name__)


def attempt(label: str, fn) -> None:
    try:
        fn()
        print(f"{label}: ok")
    except Exception as exc:  # noqa: BLE001 - probe reports every failure
        print(f"{label}: {type(exc).__name__}: {exc}")
        if label in ("Model.export", "booster.model_to_string"):
            print("---- traceback ----")
            traceback.print_exception(exc, file=sys.stdout)
            print("---- end ----")


tmp = Path(tempfile.mkdtemp())
attempt("Model.export", lambda: m.export(tmp / "exp"))
attempt("booster.model_to_string", lambda: booster.model_to_string())
attempt("pickle.dumps(booster)", lambda: pickle.dumps(booster))
attempt("joblib.dump(booster)", lambda: joblib.dump(booster, io.BytesIO()))
attempt("booster.save_model", lambda: booster.save_model(str(tmp / "b.txt")))
attempt(
    "json.dumps(pandas_categorical, default=lgb default)",
    lambda: __import__("json").dumps(
        booster.pandas_categorical,
        default=lgb.basic._json_default_with_numpy,
    ),
)
attempt(
    "json.dumps(pandas_categorical) no default",
    lambda: __import__("json").dumps(booster.pandas_categorical),
)
attempt("Model.export_code", lambda: m.export_code(tmp / "code"))
fr_no_models = dataclasses.replace(m._fit_result, models=[])
attempt("pickle FitResult(models=[])", lambda: pickle.dumps(fr_no_models))
attempt("pickle refit.pipeline_state", lambda: pickle.dumps(refit.pipeline_state))
attempt("pickle FitResult (full)", lambda: pickle.dumps(m._fit_result))
print("lightgbm", lgb.__version__)
