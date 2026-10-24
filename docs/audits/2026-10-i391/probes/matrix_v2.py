"""#391 prototype v2: refusal against an independent oracle, on this NumPy.

Oracle: the public round trip ``decode(encode(v))`` of the CURRENT tree's
codec succeeds and returns a value equal in members to ``v`` -- computed with
the prototype disabled (``refusal`` restored to the H-0133 rule while the
oracle runs). Population: the review cases, the #391 matrix, and every pair of
``SAMPLES`` values made a frozenset (and a tuple, as a control).

    PYTHONPATH=<tree> python -B docs/audits/2026-10-i391/probes/matrix_v2.py
"""

from __future__ import annotations

import datetime as dt
import decimal
import itertools
import warnings
from typing import Any

import numpy as np
import pandas as pd

import lizyml
from lizyml.core import category_values as cv
from tests.test_core.test_category_values import SAMPLES

warnings.simplefilter("ignore")
print("lizyml", lizyml.__file__, "numpy", np.__version__, "pandas", pd.__version__)
NEW = cv.refusal


def old_refusal(value: Any) -> str | None:
    return cv._refusal(value, 0)  # noqa: SLF001 -- the H-0133 rule alone


def oracle(value: Any) -> bool:
    """True when the codec writes *value* and reads the same form back."""
    cv.refusal = old_refusal  # the public codec's own checks, without the prototype
    try:
        cv.decode(cv.encode(value))
        return True
    except Exception:  # noqa: BLE001 -- any failure is "does not round-trip"
        return False
    finally:
        cv.refusal = NEW


units = ["Y", "M", "W", "D", "h", "m", "s", "ms", "us", "ns", "ps", "fs", "as", "2s", "15m"]
cases: list[tuple[str, Any]] = [
    ("review r2-1", frozenset([np.timedelta64(0, "s"), decimal.Decimal("-0")])),
    ("review r2-2", frozenset([np.uint64(9265762820973641783), np.timedelta64(4, "ms")])),
    ("safe", frozenset({4, dt.timedelta(seconds=4)})),
]
for u in units:
    for n in (0, 1, 4):
        td = np.timedelta64(n, u)
        cases.append((f"uint64({n})+td({n},{u})", frozenset({np.uint64(n), td})))
        cases.append((f"+x {u} {n}", frozenset({np.uint64(n), td, "x"})))
        cases.append((f"tuple {u} {n}", (np.uint64(n), td)))
        cases.append((f"struct {u} {n}", frozenset({(np.uint64(n),), (td,)})))
        cases.append((f"nested {u} {n}", (frozenset({np.uint64(n), td}),)))
samples = [v for vs in SAMPLES.values() for v in vs]
for i, j in itertools.combinations(range(len(samples)), 2):
    for label, build in (("fs", frozenset), ("tuple", tuple)):
        try:
            value = build([samples[i], samples[j]])
        except Exception:  # noqa: BLE001 -- not constructible on this NumPy
            continue
        cases.append((f"{label} p{i}-{j}", value))

admitted_today = 0
fires = 0
mismatch: list[str] = []
for label, value in cases:
    if old_refusal(value) is not None:
        continue  # outside V already
    admitted_today += 1
    refused = NEW(value) is not None
    fires += refused
    if refused == oracle(value):
        mismatch.append(label)
print(f"admitted today {admitted_today}, refused by prototype {fires}, oracle mismatches {len(mismatch)}")
print("fired on:", sorted(label for label, v in cases if old_refusal(v) is None and NEW(v) is not None)[:40])
print("mismatches:", mismatch[:20])

fs_pairs = [l for l, v in cases if l.startswith("fs p") and old_refusal(v) is None]
print("SAMPLES pairs as frozensets: refused", sum(NEW(v) is not None for l, v in cases if l.startswith("fs p") and old_refusal(v) is None), "of", len(fs_pairs))
print("review r2-1 refused:", NEW(cases[0][1]) is not None, "| review r2-2 refused:", NEW(cases[1][1]) is not None, "| safe refused:", NEW(cases[2][1]) is not None)
