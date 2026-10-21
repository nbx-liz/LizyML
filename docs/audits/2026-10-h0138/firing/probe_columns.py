"""H-0138: the prototype gate (h0138_rule.classify) over the identity_rows columns.

The same columns as ``identity_rows.py`` (SAMPLES singles and pairs and the
issues' own shapes, each with ``"x"``), as the training rows ``[*col, *col,
"x"]`` the probe uses, classified by the path the proposed mechanism takes.

    PYTHONPATH=.:docs/audits/2026-10-h0138/firing python docs/audits/2026-10-h0138/firing/probe_columns.py
"""

from __future__ import annotations

import decimal
import itertools
import warnings
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
from h0138_rule import classify

from tests.test_core.test_category_values import SAMPLES

warnings.simplefilter("ignore")


def _wrap(value: Any, depth: int) -> Any:
    for _ in range(depth):
        value = (value,)
    return value


samples = [v for vs in SAMPLES.values() for v in vs]
columns = [[v] for v in samples] + [list(p) for p in itertools.combinations(samples, 2)]
ones = [t(1) for t in (np.bool_, np.int8, np.int16, np.int32, np.int64,
                       np.uint8, np.uint16, np.uint32, np.uint64,
                       np.float32, np.float64, np.complex64, np.complex128)]  # fmt: skip
issue = [
    *([decimal.Decimal("1"), one] for one in ones),
    *([one, decimal.Decimal("1")] for one in ones),
    *([one, _wrap(1, d)] if first else [_wrap(1, d), one]
      for one in ones for d in (1, 2, 3, 8) for first in (True, False)),
]  # fmt: skip
tally: Counter[tuple[str, str]] = Counter()
for group, cols in (("samples", columns), ("issue shapes", issue)):
    for values in cols:
        series = pd.Series([*values, *values, "x"], dtype=object)
        tally[(group, classify(series))] += 1
print(f"pandas {pd.__version__} numpy {np.__version__}")
for (group, path), n in sorted(tally.items()):
    print(f"{n:6} {group:13} {path}")
eligible = sum(n for (_, p), n in tally.items() if p in ("faithful", "cast_raised", "predicate_false"))
fired = sum(n for (_, p), n in tally.items() if p in ("cast_raised", "predicate_false"))
print(f"build by key: {fired}/{eligible} eligible non-temporal columns")
