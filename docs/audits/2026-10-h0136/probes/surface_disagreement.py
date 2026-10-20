"""H-0136 probe: how often do Model.export and export_code disagree?

The owner asked how many cases would pass one surface and fail the other.
Real-world frequency needs user data we do not have, so this probe measures
the admitted value population instead: every sample of
``tests/test_core/test_category_values.py`` ``SAMPLES`` (the population H-0133
derives from ``ADMITTED_TYPES``) alone, and every unordered pair of them, each
in an object column beside the string ``"x"``.

For each column it fits a ``CategoricalEncoder`` (``unseen_policy="error"``)
and takes the codes of the training rows. Then, per surface:

- joblib (Model.export): the fitted encoder goes through joblib, as the saved
  fit/refit results do, and transforms the same rows.
- JSON (export_code): the learned categories go through ``encode`` /
  ``json`` / ``decode`` and the generated predict's operation runs on the same
  rows (the generated decoder is a copy a test holds to ``decode``).

A surface is "ok" when the rows transform without an error and with the
original codes. A column whose fit or original transform already fails is
counted apart (the #378 class, not a surface difference).

The counts vary with Python's hash seed (measured; the reason is not
established), so set ``PYTHONHASHSEED`` and report the range over several
seeds.

Usage: PYTHONHASHSEED=<n> python surface_disagreement.py   (from the repository root)
"""

from __future__ import annotations

import io
import itertools
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import warnings

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path.cwd()))
from lizyml.core.category_values import decode, encode  # noqa: E402
from lizyml.features.encoders.categorical_encoder import (  # noqa: E402
    CategoricalEncoder,
    comparable_values,
)
from tests.test_core.test_category_values import SAMPLES  # noqa: E402


def _codes(frame: pd.DataFrame) -> list[int]:
    return frame["c"].cat.codes.tolist()


def _joblib_ok(enc: CategoricalEncoder, df: pd.DataFrame, want: list[int]) -> bool:
    buf = io.BytesIO()
    joblib.dump(enc, buf)
    buf.seek(0)
    try:
        return _codes(joblib.load(buf).transform(df)) == want
    except Exception:  # noqa: BLE001 -- an error is the outcome measured
        return False


def _json_ok(cats: list[Any], df: pd.DataFrame, want: list[int]) -> bool:
    try:
        known = [decode(json.loads(json.dumps(encode(v)))) for v in cats]
        series, changed = comparable_values(df["c"])
        current = set(series[~changed].dropna().unique().tolist())
        if current - set(known) or changed.any():
            return False
        got = series.astype("category").cat.set_categories(known).cat.codes.tolist()
    except Exception:  # noqa: BLE001 -- an error is the outcome measured
        return False
    return bool(got == want)


def _kind(v: Any) -> str:
    tz = getattr(v, "tzinfo", None)
    zone = f"[{type(tz).__module__.split('.')[0]}]" if tz is not None else ""
    return type(v).__name__ + zone


warnings.simplefilter("ignore")  # mode-sort warnings on mixed types
samples = [v for vs in SAMPLES.values() for v in vs]
columns = [(v,) for v in samples] + list(itertools.combinations(samples, 2))
tally: Counter[str] = Counter()
kinds: dict[str, Counter[str]] = {"joblib only": Counter(), "JSON only": Counter()}
for col in columns:
    df = pd.DataFrame({"c": pd.Series([*col, *col, "x"], dtype=object)})
    try:
        enc = CategoricalEncoder(unseen_policy="error").fit(df, ["c"])
        want = _codes(enc.transform(df))
    except Exception:  # noqa: BLE001 -- counted apart
        tally["fit fails"] += 1
        continue
    j = _joblib_ok(enc, df, want)
    s = _json_ok(enc._categories["c"], df, want)
    label = {(True, True): "both ok", (False, False): "both broken"}.get(
        (j, s), "joblib only" if j else "JSON only"
    )
    tally[label] += 1
    if label in kinds:
        kinds[label][" + ".join(sorted(_kind(v) for v in col))] += 1

print(
    f"pandas={pd.__version__} numpy={np.__version__} samples={len(samples)} "
    f"PYTHONHASHSEED={os.environ.get('PYTHONHASHSEED', 'unset')}"
)
print(f"columns={len(columns)} (singletons {len(samples)}, pairs {len(columns) - len(samples)})")
for key in ("both ok", "joblib only", "JSON only", "both broken", "fit fails"):
    print(f"  {key:12s} {tally[key]}")
for label, counter in kinds.items():
    print(f"{label} (export ok, export_code broken)" if label == "joblib only" else
          f"{label} (export_code ok, export broken)")
    for kind, n in counter.most_common():
        print(f"    {n:4d}  {kind}")
