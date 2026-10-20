"""H-0136 (implementation): pandas maps a NumPy scalar equal to 1 onto a nested tuple's position.

pandas only. ``known`` holds a tuple nested 8 deep around ``1``, a NumPy
scalar equal to 1 and a string. Each line looks up targets in
``pd.Index(known)`` (what ``set_categories`` does with a batch's categories)
and casts a batch the way ``CategoricalEncoder.transform`` does. The correct
position of the scalar is 1 and of the tuple 0. The same with Python's ``True``
or ``1`` in ``known`` (what the H-0133 codec JSON gives back) is printed for
comparison, and the nesting depth is varied.

Usage: python pandas_tuple_lookup.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def nested(depth: int) -> object:
    value: object = 1
    for _ in range(depth):
        value = (value,)
    return value


def attempt(action: object) -> object:
    try:
        return action()  # type: ignore[operator]
    except Exception as exc:  # noqa: BLE001 -- the outcome measured
        return f"{type(exc).__name__}"


print(f"pandas={pd.__version__} numpy={np.__version__}")
for scalar in (np.True_, np.uint8(1), True, 1):
    for depth in (1, 2, 8):
        tup = nested(depth)
        known = pd.Index([tup, scalar, "x"], dtype=object)
        alone = attempt(lambda: known.get_indexer(pd.Index([scalar], dtype=object)).tolist())
        first = attempt(
            lambda: known.get_indexer(pd.Index([scalar, tup, "x"], dtype=object)).tolist()
        )
        batch = attempt(
            lambda: pd.Series([scalar, tup, "x"], dtype=object)
            .astype("category")
            .cat.set_categories(list(known))
            .cat.codes.tolist()
        )
        print(
            f"  {type(scalar).__module__}.{type(scalar).__name__}={scalar!r} depth {depth}: "
            f"alone -> {alone}, [scalar, tuple, x] -> {first}, batch codes -> {batch} "
            f"(correct [1, 0, 2])"
        )
