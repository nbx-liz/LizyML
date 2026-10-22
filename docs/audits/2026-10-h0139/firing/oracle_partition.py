"""Count H-0139's Y and R overlay sets in the H-0138 oracle of this pandas.

Y: original temporal cases whose input samples hold a timedelta64 in Y or M
units (refused at fit under H-0139); case ids ``s<i>`` / ``p<i>-<j>`` index
the H-0133 SAMPLES as ``make_oracle.py`` enumerates them in this environment.
R: original temporal cases that failed at the cast, excluding Y (rebuilt by
key under H-0139). The third set, S, names cases whose answer changes without
either; it needs the implementation and is not counted here.

    PYTHONPATH=.:docs/audits/2026-10-h0139/firing python -B docs/audits/2026-10-h0139/firing/oracle_partition.py
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pandas as pd
from h0139_rule import calendar_units

from tests.test_core.test_category_values import SAMPLES

samples = [v for vs in SAMPLES.values() for v in vs]
calendar = {i for i, v in enumerate(samples) if calendar_units(v)}
path = Path(f"tests/test_features/data/h0138_oracle_pandas-{pd.__version__}.json.gz")
cases = json.loads(gzip.decompress(path.read_bytes()))["temporal_cases"]


def _indices(case_id: str) -> set[int]:
    return {int(i) for i in case_id[1:].split("-")}


y = [c["id"] for c in cases if _indices(c["id"]) & calendar]
r = [
    c["id"]
    for c in cases
    if not (_indices(c["id"]) & calendar) and c["outcome"].get("fails_at") == "cast"
]
print(f"pandas {pd.__version__}: calendar sample indices {sorted(calendar)}")
print(f"  temporal cases {len(cases)}  Y {len(y)}  R {len(r)}")
