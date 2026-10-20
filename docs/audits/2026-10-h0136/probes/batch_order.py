"""H-0136 (implementation): does pandas give one value different codes in different batches?

pandas only, no LizyML and no serialization. For a learned category list
holding a deeply nested tuple and a NumPy scalar equal to 1, each batch is
cast with ``astype("category")`` and mapped with ``set_categories(known)``,
as ``CategoricalEncoder.transform`` and the generated ``predict.py`` do. The
same is repeated with the JSON projection of ``known`` (NumPy scalars become
Python values; the tuple stays).

Usage: python batch_order.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

NESTED = ((((((((1,),),),),),),),)


def codes(rows: list[object], known: list[object]) -> list[int]:
    series = pd.Series(rows, dtype=object).astype("category")
    return series.cat.set_categories(known).cat.codes.tolist()


print(f"pandas={pd.__version__} numpy={np.__version__}")
for scalar, plain in ((np.True_, True), (np.uint8(1), 1)):
    name = type(scalar).__name__
    for label, known in (("original", [NESTED, scalar, "x"]), ("JSON", [NESTED, plain, "x"])):
        print(f"{name} {label} known={known}")
        for batch_name, rows in (
            (f"[{name}, nested, x]", [scalar, NESTED, "x"]),
            (f"[nested, {name}, x]", [NESTED, scalar, "x"]),
            (f"[{name}] alone", [scalar]),
            ("[nested] alone", [NESTED]),
        ):
            try:
                result: object = codes(rows, known)
            except Exception as exc:  # noqa: BLE001 -- the outcome measured
                result = f"{type(exc).__name__}: {exc}"
            print(f"    {batch_name:22s} -> {result}")
