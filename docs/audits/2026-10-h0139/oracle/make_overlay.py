"""H-0139: the closed list of H-0138 temporal oracle cases H-0139 changes on purpose.

The three H-0138 oracle files stay byte-for-byte. This script writes, for the
running pandas version, ``tests/test_features/data/h0139_overlay_pandas-<v>.json.gz``:
the ids, each one's frozen ``before`` outcome and its ``after`` outcome under
H-0139. Membership and ``after`` come from the H-0138 oracle file and the
prototype rule (``docs/audits/2026-10-h0139/firing/h0139_rule.py``) only;
the library under change is never run, so the list cannot be chosen by
looking at the implementation's answers (H-0139 mechanism 7).

The three sets (HISTORY.md ``## H-0139``, 対応方針 7):

- ``Y``: cases whose input samples hold a timedelta64 in Y or M units ->
  refused by the fit validation;
- ``R``: the remaining cases whose frozen outcome failed at the cast -> built
  by key (stand-in);
- ``S``: (a) cases the prototype rule finds not faithful -> built by key;
  (b) cases whose frozen codes disagree with the key positions of their frozen
  categories -> the same categories, codes by key (transform-only).

Built by key: the first value of each key over the training rows
``[*values, *values, "x"]``, ordered by ``str``; codes are key positions of
the frozen batch ``[*distinct(values), "x"]`` and its reverse.

    PYTHONHASHSEED=0 PYTHONPATH=.:docs/audits/2026-10-h0139/firing \
        python -B docs/audits/2026-10-h0139/oracle/make_overlay.py
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from h0139_rule import calendar_units, category_key, classify

from lizyml.core.category_values import decode
from tests.test_features import _h0138_oracle as h

warnings.simplefilter("ignore")
OUT = Path("tests/test_features/data")


def _rebuilt(values: list[Any]) -> dict[str, Any]:
    train = [*values, *values, "x"]
    reps: dict[Any, Any] = {}
    for value in train:
        reps.setdefault(category_key(value), value)
    cats = sorted(reps.values(), key=str)
    position = {category_key(c): i for i, c in enumerate(cats)}
    batch = [*h.distinct(values), "x"]
    forward = [position[category_key(v)] for v in batch]
    return {
        "categories": [h.typed(v) for v in cats],
        "stand_in": ["c"],
        "codes_forward": forward,
        "codes_reversed": forward[::-1],
    }


def _keyed_codes(before: dict[str, Any], values: list[Any]) -> list[int] | None:
    """Key positions of the frozen batch in the frozen categories, or None."""
    cats = [decode(c["value"]) for c in before["categories"]]
    position = {category_key(c): i for i, c in enumerate(cats)}
    if len(position) != len(cats):
        return None
    batch = [*h.distinct(values), "x"]
    keys = [category_key(v) for v in batch]
    if any(k not in position for k in keys):
        return None
    return [position[k] for k in keys]


def main() -> None:
    assert os.environ.get("PYTHONHASHSEED") == "0", "run under PYTHONHASHSEED=0"
    oracle = h.oracle()
    changes: list[dict[str, Any]] = []
    sets: dict[str, list[str]] = {"Y": [], "R": [], "S": []}
    unchanged_failures: Counter[str] = Counter()
    for case in oracle["temporal_cases"]:
        cid, before = case["id"], case["outcome"]
        values = h.probe_values(cid)
        units = sorted(set().union(*(calendar_units(v) for v in values)))
        if units:
            rule, after = "Y", {
                "fails_at": "fit_validation",
                "exception": "LizyMLError",
                "code": "DATA_SCHEMA_INVALID",
                "context": {"column": "c", "units": units, "reason": "calendar_timedelta64"},
            }
        elif before.get("fails_at") == "cast":
            rule, after = "R", _rebuilt(values)
        elif "fails_at" in before:
            unchanged_failures[before["fails_at"]] += 1
            continue
        else:
            train = pd.Series([*values, *values, "x"], dtype=object)
            if classify(train)["path"] != "faithful":
                rule, after = "S", _rebuilt(values)
            else:
                forward = _keyed_codes(before, values)
                assert forward is not None, f"{cid}: faithful but not keyable"
                if [forward, forward[::-1]] == [
                    before["codes_forward"],
                    before["codes_reversed"],
                ]:
                    continue
                rule = "S"
                after = {**before, "codes_forward": forward, "codes_reversed": forward[::-1]}
        sets[rule].append(cid)
        changes.append({"id": cid, "rule": rule, "before": before, "after": after})
    hashed = [{"id": c["id"], "before": c["before"], "after": c["after"]} for c in changes]
    digest = hashlib.sha256(
        json.dumps(hashed, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    meta = {
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "temporal_cases": len(oracle["temporal_cases"]),
        "counts": {k: len(v) for k, v in sets.items()},
        "S": sets["S"],
        "changed_sha256": digest,
        "generator": "docs/audits/2026-10-h0139/oracle/make_overlay.py",
    }
    path = OUT / f"h0139_overlay_pandas-{pd.__version__}.json.gz"
    payload = json.dumps({"meta": meta, "changes": changes}, sort_keys=True)
    path.write_bytes(gzip.compress(payload.encode(), mtime=0))
    print(f"pandas {pd.__version__}: {meta['counts']} total {len(changes)}")
    print(f"  S {sets['S']}")
    print(f"  unchanged failures {dict(unchanged_failures)}")
    print(f"  changed sha256 {digest}")


if __name__ == "__main__":
    main()
