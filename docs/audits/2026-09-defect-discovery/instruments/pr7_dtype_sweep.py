"""#267: does any ordinary column x target dtype pair raise inside the guarded call?

Columns: the 33 dtypes of PR 6's sweep (``tests/test_features/test_column_dtype_check.py``
``_DTYPES``), each built twice -- equal to the target's values and not. Targets:
int64, float64, bool, Int64, object strings, category, string. A cell "raises"
when ``_series_perfectly_correlated`` raises; those are the cells the handler
swallows today and that would propagate once it is removed.
"""

from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

from lizyml.data.validators import _series_perfectly_correlated  # noqa: E402
from tests.test_features import test_column_dtype_check as m  # noqa: E402

base = m._VALS
targets = {
    "int64": base.astype("int64"),
    "float64": base.astype("float64"),
    "bool": (base % 2).astype(bool),
    "Int64": base.astype("Int64"),
    "object_str": base.astype(str).astype(object),
    "category": base.astype("category"),
    "string": base.astype(str).astype("string"),
}
raised = 0
cells = 0
for cname, build in sorted(m._DTYPES.items()):
    col = build()
    for variant, c in (("equal", col), ("shifted", col.iloc[::-1].reset_index(drop=True))):
        for tname, y in targets.items():
            cells += 1
            try:
                _series_perfectly_correlated(c, y)
            except (TypeError, ValueError) as exc:
                raised += 1
                print(f"RAISES col={cname:16s} {variant:7s} target={tname:10s} {type(exc).__name__}: {exc}")
            except Exception as exc:  # noqa: BLE001 -- not caught by the handler either
                print(f"OTHER  col={cname:16s} {variant:7s} target={tname:10s} {type(exc).__name__}: {exc}")
print(f"cells raising TypeError/ValueError: {raised}/{cells}")
_ = np
