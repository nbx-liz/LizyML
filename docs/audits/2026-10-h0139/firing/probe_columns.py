"""H-0139: the prototype rule (h0139_rule.classify) over the H-0138 probe columns.

The same columns as ``docs/audits/2026-10-h0138/firing/probe_columns.py``
(SAMPLES singles and pairs and the H-0138 issue shapes, each beside ``"x"``,
as the training rows ``[*col, *col, "x"]``), classified by the path the
H-0139 mechanism takes. Reproduces the activation counts of Codex run
``h0139-mechanism-design`` (finding 7) from a committed script.

    PYTHONPATH=.:docs/audits/2026-10-h0139/firing python -B docs/audits/2026-10-h0139/firing/probe_columns.py
"""

from __future__ import annotations

import decimal
import itertools
import warnings
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
from h0139_rule import classify

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
errors = 0
for values in columns + issue:
    series = pd.Series([*values, *values, "x"], dtype=object)
    try:
        out = classify(series)
    except Exception as exc:  # noqa: BLE001 -- counted, never skipped
        errors += 1
        print("rule_error", type(exc).__name__, str(exc)[:120])
        continue
    tally[("temporal" if out["temporal"] else "non-temporal", out["path"])] += 1
print(f"pandas {pd.__version__} numpy {np.__version__} columns {len(columns) + len(issue)}")
for (group, path), n in sorted(tally.items()):
    print(f"{n:6} {group:13} {path}")
temporal = sum(n for (g, _), n in tally.items() if g == "temporal")
fired = sum(
    n
    for (g, p), n in tally.items()
    if p == "ym_refused" or (g == "temporal" and p in ("cast_raised", "predicate_false"))
)
print(f"H-0139 activation: {fired}/{temporal} temporal columns; rule errors {errors}")
