"""H-0138: which columns change under each candidate category identity?

For every column made of one or two values from the H-0133 ``SAMPLES`` (plus
the string ``"x"``), this compares:

- **now**: what the library does today. The training rows go through
  ``astype("category")`` (as ``dataframe_builder._apply_categorical`` does)
  and ``CategoricalEncoder(unseen_policy="error").fit``; then the distinct
  values are transformed in two batch orders. The column is *correct now*
  when fit and both transforms succeed and every value gets the same code in
  both orders.
- **A** (typed): the codec's saved form, ``json.dumps(encode(v))``. NumPy
  plain scalars become their Python value (``np.int8(-3)`` is ``-3``), and
  ``bool``, ``int``, ``float``, ``Decimal``, ``complex`` stay apart.
- **B** (numeric): A, but numbers that are equal as numbers are one value
  (``-0.0``, ``0``, ``False`` and ``Decimal("-0")``), as Python's ``==`` says.

It prints one line per column whose distinct-value count differs between
the three, or that is not correct now, and a tally by the pair's types.
Temporal values are tallied too, but H-0138 does not decide them (H-0139).

    python docs/audits/2026-10-h0138/identity_rows.py   (repository root on the path)
"""

from __future__ import annotations

import decimal
import fractions
import itertools
import json
import math
import sys
import warnings
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

from lizyml.core.category_values import encode
from lizyml.features.encoders.categorical_encoder import CategoricalEncoder
from tests.test_core.test_category_values import SAMPLES

warnings.simplefilter("ignore")
TEMPORAL = ("date", "datetime", "timedelta", "Timestamp", "Timedelta", "datetime64", "timedelta64")


def key_a(value: Any) -> str:
    return json.dumps(encode(value), sort_keys=True)


def _number(value: Any) -> Any:
    """An exact number for bool/int/float/Decimal/real complex, else None."""
    if isinstance(value, (np.datetime64, np.timedelta64)):  # numpy integer subclasses
        return None
    if isinstance(value, (complex, np.complexfloating)):
        c = complex(value)
        return _number(c.real) if c.imag == 0 else None
    if isinstance(value, (bool, int, np.bool_, np.integer)):
        return fractions.Fraction(int(value))
    if isinstance(value, (float, np.floating)):
        f = float(value)
        return ("inf", math.copysign(1, f)) if math.isinf(f) else fractions.Fraction(f)
    if type(value) is decimal.Decimal:
        if value.is_infinite():
            return ("inf", -1.0 if value.is_signed() else 1.0)
        return fractions.Fraction(value)
    return None


def key_b(value: Any) -> str:
    n = _number(value)
    if n is not None:
        return "num:" + repr(n)
    if type(value) is tuple:
        return "tuple:" + "|".join(key_b(v) for v in value)
    if type(value) is frozenset:
        return "frozenset:" + "|".join(sorted(key_b(v) for v in value))
    return key_a(value)


def now(values: list[Any]) -> tuple[str, int | None]:
    """('ok' | an outcome, the number of learned categories)."""
    distinct: list[Any] = []  # by identity: equal values may be different objects
    for v in values:
        if not any(v is d for d in distinct):
            distinct.append(v)
    train = pd.DataFrame({"c": pd.Series([*values, *values, "x"], dtype=object)})
    try:
        train["c"] = train["c"].astype("category")
    except Exception as exc:  # noqa: BLE001 -- the fit cast fails: an outcome
        return f"cast {type(exc).__name__}", None
    try:
        enc = CategoricalEncoder(unseen_policy="error").fit(train, ["c"])
    except Exception as exc:  # noqa: BLE001 -- fit fails: an outcome
        return f"fit {type(exc).__name__}", None
    n = len(enc.get_state()["categories"]["c"])
    codes = []
    for batch in ([*distinct, "x"], ["x", *reversed(distinct)]):
        frame = pd.DataFrame({"c": pd.Series(batch, dtype=object)})
        try:
            out = enc.transform(frame)["c"].cat.codes.tolist()
        except Exception as exc:  # noqa: BLE001 -- transform fails: an outcome
            return f"transform {type(exc).__name__}", n
        codes.append(dict(zip(map(id, batch), out, strict=True)))
    if codes[0] != codes[1]:
        return "batch-order codes", n
    return "ok", n


def _wrap(value: Any, depth: int) -> Any:
    for _ in range(depth):
        value = (value,)
    return value


def kind(values: list[Any]) -> str:
    return " + ".join(sorted(f"{type(v).__module__}.{type(v).__qualname__}" for v in values))


samples = [v for vs in SAMPLES.values() for v in vs]
columns = [[v] for v in samples] + [list(p) for p in itertools.combinations(samples, 2)]
# The issues' own shapes, which SAMPLES does not hold: #378 (a Decimal and an
# equal NumPy integer) and #380 (a NumPy scalar equal to 1 beside 1 wrapped in
# tuples), in both orders (Codex critique run v021-category-critique).
_NP_ONES = [t(1) for t in (np.bool_, np.int8, np.int16, np.int32, np.int64,
                           np.uint8, np.uint16, np.uint32, np.uint64,
                           np.float32, np.float64, np.complex64, np.complex128)]  # fmt: skip
EXTRA = [
    *([decimal.Decimal("1"), one] for one in _NP_ONES),
    *([one, decimal.Decimal("1")] for one in _NP_ONES),
    *(
        [one, nested] if first else [nested, one]
        for one in _NP_ONES
        for depth in (1, 2, 3, 8)
        for nested in [_wrap(1, depth)]
        for first in (True, False)
    ),
]
columns += EXTRA
print(f"python {sys.version.split()[0]} pandas {pd.__version__} numpy {np.__version__}")
print(f"{len(columns)} columns ({len(samples)} singles, their pairs and {len(EXTRA)} issue shapes), each with 'x'")
tally: Counter[tuple[str, str]] = Counter()
lines = []
for values in columns:
    with_x = [*values, "x"]
    outcome, n_now = now(values)
    n_a = len({key_a(v) for v in with_x})
    n_b = len({key_b(v) for v in with_x})
    temporal = any(type(v).__name__ in TEMPORAL for v in values)
    if outcome != "ok" or n_now != n_a or n_a != n_b:
        label = (
            f"now={outcome}:{n_now} A={n_a} B={n_b}"
            + (" [temporal]" if temporal else "")
        )
        tally[(label.split(" [")[0].split(":")[0] + (" temporal" if temporal else ""), kind(values))] += 1
        lines.append(f"{label:58} {kind(values)}  {values!r}"[:300])
for line in lines:
    print(line)
print("--- tally (outcome now, temporal flag) x pair types ---")
for (label, k), c in sorted(tally.items()):
    print(f"{c:4} {label:40} {k}")
