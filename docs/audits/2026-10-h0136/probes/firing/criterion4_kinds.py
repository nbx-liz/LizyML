"""H-0136: which exception (or code change) each JSON-breaking pair of the grid gives.

The Proposal's refusal ``kind`` must name what really happens, so this probe
runs the prototype rule on every column of the ``surface_disagreement.py``
grid that breaks under JSON only and tallies the raw outcome (the exception
type, or ``codes`` / ``mode``) by the pair's types.

Usage: PYTHONHASHSEED=<n> python criterion4_kinds.py   (this directory and the repository on the path)
"""

from __future__ import annotations

import itertools
import warnings
from collections import Counter
from typing import Any

import pandas as pd
from h0136_rule import state_breaks

from lizyml.features.encoders.categorical_encoder import CategoricalEncoder
from tests.test_core.test_category_values import SAMPLES

warnings.simplefilter("ignore")
samples = [v for vs in SAMPLES.values() for v in vs]
tally: Counter[tuple[str, str, str]] = Counter()
for col in [(v,) for v in samples] + list(itertools.combinations(samples, 2)):
    df = pd.DataFrame({"c": pd.Series([*col, *col, "x"], dtype=object)})
    try:
        state = {"encoder": CategoricalEncoder().fit(df, ["c"]).get_state()}
        breaks, _ = state_breaks(state)
    except Exception:  # noqa: BLE001 -- the fit itself fails: not counted
        continue
    for b in breaks:
        kinds: Any = " + ".join(sorted(type(v).__name__ for v in col))
        tally[(b["serializer"], b["kind"], kinds)] += 1
import os  # noqa: E402

totals = Counter()
for (serializer, _, _), n in tally.items():
    totals[serializer] += n
print(
    f"pandas={pd.__version__} PYTHONHASHSEED={os.environ.get('PYTHONHASHSEED', 'unset')} "
    f"breaks by serializer: {dict(sorted(totals.items()))}"
)
for (serializer, kind, types), n in sorted(tally.items()):
    print(f"  {serializer:6s} {kind:16s} {n:4d}  {types}")
