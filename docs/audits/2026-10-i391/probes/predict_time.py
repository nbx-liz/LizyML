"""#391: a keyed column receives the collapsing frozenset only at predict.

Fit tiny models on ordinary category values (no frozenset), then predict rows
holding ``frozenset({np.uint64(4), np.timedelta64(4, "s")})`` and a control
unseen value. Prints the encoder's identity columns and, per policy, what
predict does: the codes / predictions, or the exception.

    PYTHONPATH=<tree> python -B docs/audits/2026-10-i391/probes/predict_time.py
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

import lizyml
from lizyml import Model
from lizyml.core.category_values import refusal
from tests._helpers import make_config

warnings.simplefilter("ignore")
V = frozenset({np.uint64(4), np.timedelta64(4, "s")})
print("lizyml", lizyml.__file__, "numpy", np.__version__, "pandas", pd.__version__)
print("refusal(V):", refusal(V))


def model_on(values: list[Any], policy: str) -> tuple[Model, pd.DataFrame]:
    rng = np.random.default_rng(0)
    rows = [*values, "x"] * 20
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


for values in (["a", "b"], [np.uint64(5), "a"], [(1,), frozenset({2, 3})]):
    for policy in ("mode", "nan", "error"):
        try:
            model, df = model_on(values, policy)
        except Exception as exc:  # noqa: BLE001 -- the probe reports it
            print(f"{values!r:40} {policy:5} fit raised {type(exc).__name__}: {exc}"[:240])
            continue
        state = model._refit_result.pipeline_state  # noqa: SLF001
        enc = state["encoder"]
        ids = enc.get("identity_columns") if isinstance(enc, dict) else None
        frame = df.drop(columns=["target"]).head(3).copy()
        out = {}
        for label, value in (("collapsing", V), ("unseen", "never-seen")):
            probe = frame.copy()
            probe["c"] = pd.Series([value, values[0], "x"], dtype=object, index=probe.index)
            try:
                pred = model.predict(probe).pred
                out[label] = f"ok {np.round(pred, 4).tolist()}"
            except Exception as exc:  # noqa: BLE001 -- the probe reports it
                out[label] = f"{type(exc).__name__}: {str(exc)[:90]}"
        print(f"{values!r:40} {policy:5} identity_columns={ids} -> {out}")
