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
- ``T`` (H-0144): the remaining cases whose frozen outcome failed at the
  transform (pandas' cast held the column, the old lookup failed) -> built by
  key in the cast's category order (H-0138: a declared column keeps its
  order). A rule with no case is left out of ``meta["counts"]``, so the
  overlays written before T existed keep their bytes.

H-0144: the key and the codec come from a frozen copy of
``lizyml/core/{category_values,category_temporal,exceptions}.py``
(``docs/audits/2026-10-i389/oracle/frozen_core/``, taken at ``f025620``),
served by an import hook that refuses every other ``lizyml`` module, so an
overlay never follows the library it checks.

Built by key: the first value of each key over the training rows
``[*values, *values, "x"]``, ordered by ``str``; codes are key positions of
the frozen batch ``[*distinct(values), "x"]`` and its reverse.

    PYTHONHASHSEED=0 PYTHONPATH=.:docs/audits/2026-10-h0139/firing \
        python -B docs/audits/2026-10-h0139/oracle/make_overlay.py
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import importlib.util
import sys
from pathlib import Path

_FROZEN = (
    Path(__file__).resolve().parents[2] / "2026-10-i389" / "oracle" / "frozen_core"
)
_SERVED = {
    "lizyml.core.category_values": "category_values",
    "lizyml.core.category_temporal": "category_temporal",
    "lizyml.core.exceptions": "exceptions",
}


class _FrozenCore(importlib.abc.MetaPathFinder):
    """Serve the frozen codec under its own names; refuse any other lizyml."""

    def find_spec(self, name, path, target=None):  # type: ignore[no-untyped-def]
        if name in ("lizyml", "lizyml.core"):
            spec = importlib.machinery.ModuleSpec(name, None, is_package=True)
            spec.submodule_search_locations = []
            return spec
        if name in _SERVED:
            return importlib.util.spec_from_file_location(
                name, _FROZEN / f"{_SERVED[name]}.py"
            )
        if name.startswith("lizyml."):
            raise ImportError(f"the overlay generator does not import {name}")
        return None


assert not any(m == "lizyml" or m.startswith("lizyml.") for m in sys.modules)
sys.meta_path.insert(0, _FrozenCore())

import gzip  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import warnings  # noqa: E402
from collections import Counter  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from h0139_rule import calendar_units, category_key, classify  # noqa: E402

from lizyml.core.category_values import decode  # noqa: E402
from tests.test_features import _h0138_oracle as h  # noqa: E402

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


def _rebuilt_cast_order(values: list[Any]) -> dict[str, Any]:
    """Rule T: the cast's categories in their order, then the training rows,
    deduplicated by key (the first value of each key); codes by key."""
    train = pd.Series([*values, *values, "x"], dtype=object)
    cast = list(train.astype("category").cat.categories)
    reps: dict[Any, Any] = {}
    for value in [*cast, *train.tolist()]:
        reps.setdefault(category_key(value), value)
    cats = list(reps.values())
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
    sets: dict[str, list[str]] = {"Y": [], "R": [], "S": [], "T": []}
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
        elif before.get("fails_at") == "transform":
            rule, after = "T", _rebuilt_cast_order(values)
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
        "counts": {k: len(v) for k, v in sets.items() if v},
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
