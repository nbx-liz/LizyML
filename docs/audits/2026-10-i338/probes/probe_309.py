"""#309 concrete example: integer categories at fit, float arrivals at predict."""

import warnings

import numpy as np
import pandas as pd

from lizyml.core.model import Model
from tests._helpers import make_config

rng = np.random.default_rng(0)
n = 120
vals = pd.Series(rng.integers(0, 5, n))
b = rng.normal(size=n)
y = ((vals.to_numpy() + b) > 2).astype(int)
df = pd.DataFrame({"a": pd.Series(list(vals), dtype=object), "b": b, "target": y})
model = Model(make_config("binary"), data=df)
model.fit()


def show(label: str, a: pd.Series) -> None:
    X = pd.DataFrame({"a": a, "b": [0.0] * len(a)})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            r = model.predict(X)
        except Exception as e:  # noqa: BLE001 - the probe reports what escapes
            print(f"{label:<14} {type(e).__name__}: {str(e)[:100]}")
            return
    print(f"{label:<14} proba={np.round(r.proba, 3).tolist()}")
    for m in r.warnings:
        print(f"{'':<14} warning: {m[:140]}")


ints = [0, 1, 2, 3, 4]
show("int64", pd.Series(ints, dtype="int64"))
show("float64", pd.Series(ints, dtype="float64"))
show("float32", pd.Series(ints, dtype="float32"))
show("float16", pd.Series(ints, dtype="float16"))
show("longdouble", pd.Series(ints, dtype=np.longdouble))
print("--- a value that is not a fit category")
show("float64 2.5", pd.Series([2.5], dtype="float64"))
show("float16 2.5", pd.Series([2.5], dtype="float16"))
