"""#309: arrivals the dtype matrix does not cover -- numpy float16 / longdouble
scalars inside an ``object`` column, and a ``category`` column built from them.

Runs the current ``CategoricalEncoder`` fitted on integer categories 0..4.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from lizyml.features.encoders.categorical_encoder import CategoricalEncoder

enc = CategoricalEncoder("mode").fit(pd.DataFrame({"a": pd.Series([0, 1, 2, 3, 4], dtype=object)}), ["a"])


def run(label: str, build) -> None:  # noqa: ANN001
    try:
        s = build()
    except Exception as exc:  # noqa: BLE001 - the probe reports what escapes
        print(f"{label:<34} build: {type(exc).__name__}: {exc}")
        return
    try:
        out, warns = enc.transform_with_warnings(pd.DataFrame({"a": s}))
        print(f"{label:<34} ok codes={out['a'].cat.codes.tolist()} warnings={len(warns)}")
    except Exception as exc:  # noqa: BLE001
        print(f"{label:<34} transform: {type(exc).__name__}: {exc}")


run("object of np.float16", lambda: pd.Series([np.float16(v) for v in (0, 1, 2)], dtype=object))
run("object of np.float16 + unseen", lambda: pd.Series([np.float16(v) for v in (0, 1, 2.5)], dtype=object))
run("object of np.longdouble", lambda: pd.Series([np.longdouble(v) for v in (0, 1, 2)], dtype=object))
run("category of np.longdouble", lambda: pd.Series(pd.Categorical([np.longdouble(v) for v in (0, 1, 2)])))
run("category of float16 array", lambda: pd.Series(pd.Categorical(np.array([0, 1, 2], dtype=np.float16))))
