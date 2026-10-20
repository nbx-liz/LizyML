"""H-0136: where the prototype rule and the grid's JSON check disagree, ask the real generated predict.py.

``surface_disagreement.py`` judges the JSON side with its own copy of the
generated transform; ``h0136_rule.py`` uses the library encoder on the
projected state. For every grid column where the two disagree, this probe
fits a real model on that column, runs ``export_code`` and calls the
generated ``predict.py``'s ``transform`` on the training rows, comparing
codes with the library's in-memory transform.

Usage: PYTHONHASHSEED=<n> python oracle_disagreement.py   (this directory and the repository on the path)
"""

from __future__ import annotations

import importlib.util
import itertools
import tempfile
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from h0136_rule import state_breaks

from lizyml import Model
from lizyml.features.encoders.categorical_encoder import CategoricalEncoder
from tests.test_core.test_category_values import SAMPLES

warnings.simplefilter("ignore")
sys_path = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("grid", sys_path / "surface_disagreement.py")
assert spec is not None and spec.loader is not None
# Import only the helpers: the module runs its grid at import, so read its source instead.
src = (sys_path / "surface_disagreement.py").read_text(encoding="utf-8")
helpers: dict[str, Any] = {}
exec(src.split("warnings.simplefilter")[0], helpers)  # noqa: S102 -- our own probe file
json_ok = helpers["_json_ok"]
codes = helpers["_codes"]

samples = [v for vs in SAMPLES.values() for v in vs]
disagree = []
for col in [(v,) for v in samples] + list(itertools.combinations(samples, 2)):
    df = pd.DataFrame({"c": pd.Series([*col, *col, "x"], dtype=object)})
    try:
        enc = CategoricalEncoder(unseen_policy="error").fit(df, ["c"])
        want = codes(enc.transform(df))
    except Exception:  # noqa: BLE001
        continue
    grid_ok = json_ok(enc._categories["c"], df, want)
    rule_ok = not [b for b in state_breaks({"encoder": enc.get_state()})[0] if b["serializer"] == "json"]
    if grid_ok != rule_ok:
        disagree.append((col, grid_ok, rule_ok))

print(f"pandas={pd.__version__} disagreements={len(disagree)}")
for col, grid_ok, rule_ok in disagree:
    rng = np.random.default_rng(0)
    values = list(col) + ["x"]
    idx = rng.integers(0, len(values), 400)
    idx[: len(values)] = np.arange(len(values))
    frame = pd.DataFrame(
        {
            "c": pd.Series([values[i] for i in idx], dtype=object),
            "num": rng.normal(size=400),
            "target": (idx % 2 == 0).astype(int),
        }
    )
    cfg = {
        "config_version": 1,
        "task": "binary",
        "data": {"target": "target"},
        "features": {"categorical": ["c"], "unseen_policy": "error"},
        "split": {"method": "kfold", "n_splits": 2, "random_state": 0},
        "model": {"name": "lgbm", "params": {"n_estimators": 5, "verbose": -1}},
    }
    X = frame.drop(columns=["target"])
    try:
        model = Model(cfg)
        model.fit(data=frame)
        out = Path(tempfile.mkdtemp()) / "code"
        model.export_code(out)
        mspec = importlib.util.spec_from_file_location("gen", out / "predict.py")
        assert mspec is not None and mspec.loader is not None
        gen = importlib.util.module_from_spec(mspec)
        mspec.loader.exec_module(gen)
        got = gen.transform(X)["c"]
        ref = model._refit_result.pipeline_state  # type: ignore[union-attr]
        lib = CategoricalEncoder().load_state({**ref["encoder"], "unseen_policy": "error"})
        expected = lib.transform(X[["c"]])["c"].cat.codes.to_numpy(dtype=float)
        expected[expected < 0] = np.nan
        same = np.array_equal(np.asarray(got, dtype=float), expected, equal_nan=True)
        real = "generated transform matches" if same else "generated codes DIFFER"
    except Exception as exc:  # noqa: BLE001 -- the outcome measured
        real = f"{type(exc).__name__}: {str(exc).splitlines()[0][:90]}"
    shown = [f"{type(v).__name__}({v!r})"[:40] for v in col]
    print(f"  {shown} grid_ok={grid_ok} rule_ok={rule_ok} -> {real}")
