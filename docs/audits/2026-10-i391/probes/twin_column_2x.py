"""H-0145 acceptance 6b on NumPy 2.x: a column holding the large-uint64 value
and its key-equal twin. Prints the builder's path, the cast categories, the
encoder's identity columns, the fit warnings, the export outcome and the
predictions of the value against its twin.

    PYTHONPATH=<tree> python -B docs/audits/2026-10-i391/probes/twin_column_2x.py <scratch>
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from lizyml.core import category_values as cv
from lizyml.core.specs.feature_spec import FeatureSpec
from lizyml.data.dataframe_builder import _apply_categorical
from tests.test_core.test_h0145_round_trip import CASES
from tests.test_features.test_h0145_fit_and_predict import _encoder, _fit, _predict_with

print("numpy", np.__version__, "pandas", pd.__version__)
for name, (v, twin) in CASES.items():
    print(f"== {name}: refusal(v)={cv.refusal(v)!r}")
    try:
        eq: object = v == twin
    except Exception as exc:  # noqa: BLE001 -- the probe reports it
        eq = type(exc).__name__
    print(f"   v == twin: {eq}, hash equal: {hash(v) == hash(twin)}")
    col = pd.Series([twin, twin, v, "x"] * 5, dtype=object)
    print("   builder eligible:", cv.identity_eligible(col))
    built = _apply_categorical(pd.DataFrame({"c": col}), FeatureSpec(categorical=("c",)))
    cats = list(built["c"].cat.categories) if hasattr(built["c"], "cat") else ["<object column>"]
    print("   cast categories:", [sorted(map(repr, c)) if isinstance(c, frozenset) else c for c in cats])
    model, caught = _fit([twin, twin, v, "x"])
    print("   identity_columns:", _encoder(model)["identity_columns"])
    print("   warnings:", [str(w.message)[:80] for w in caught if "saved model" in str(w.message)])
    target = Path(sys.argv[1]) / name
    try:
        model.export(target)
        print("   export: written")
    except Exception as exc:  # noqa: BLE001 -- the probe reports it
        print("   export:", type(exc).__name__, str(exc)[:100])
    print("   predict v:", _predict_with(model, v).tolist(), "twin:", _predict_with(model, twin).tolist())
    warnings.resetwarnings()
