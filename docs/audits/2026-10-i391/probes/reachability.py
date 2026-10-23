"""#391 acceptance 6: on the tree before the change, a column holding the
collapsing frozenset cannot produce a saved artifact on NumPy 1.24.

Fits ``[V, "x"]`` (``V = frozenset({np.uint64(4), np.timedelta64(4, "s")})``),
then tries ``Model.export`` and ``export_code`` into scratch directories and
reports what was raised and whether anything was written.

    PYTHONPATH=<tree> python -B docs/audits/2026-10-i391/probes/reachability.py <scratch>
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import lizyml
from lizyml import Model
from lizyml.core.category_values import refusal
from lizyml.core.exceptions import LizyMLError
from tests._helpers import make_config

warnings.simplefilter("ignore")
V = frozenset({np.uint64(4), np.timedelta64(4, "s")})
scratch = Path(sys.argv[1])
print("lizyml", lizyml.__file__, "numpy", np.__version__, "pandas", pd.__version__)
print("refusal(V):", refusal(V))

rng = np.random.default_rng(0)
rows = [V, "x"] * 20
df = pd.DataFrame(
    {
        "c": pd.Series(rows, dtype=object),
        "num": rng.normal(size=len(rows)),
        "target": rng.normal(size=len(rows)),
    }
)
model = Model(make_config("regression", n_estimators=5, n_splits=2))
try:
    model.fit(data=df)
    print("fit: returned")
except LizyMLError as exc:
    print(f"fit: {exc.code.value} {str(exc)[:120]}")
for name, call in (
    ("Model.export", lambda p: model.export(p)),
    ("export_code", lambda p: model.export_code(p)),
):
    target = scratch / name.replace(".", "_")
    try:
        call(target)
        print(f"{name}: returned")
    except LizyMLError as exc:
        print(f"{name}: {exc.code.value} {str(exc)[:120]}")
    except Exception as exc:  # noqa: BLE001 -- the probe reports it
        print(f"{name}: {type(exc).__name__} {str(exc)[:120]}")
    written = sorted(str(p.relative_to(scratch)) for p in target.rglob("*")) if target.exists() else []
    print(f"  written: {written}")
