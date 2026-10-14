"""Q2 supplement: object-dtype (not declared category) columns of these values."""

import tempfile
import warnings
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

from lizyml import Model
from tests._helpers import make_config

warnings.filterwarnings("ignore")
rng = np.random.default_rng(0)
n = 120
idx = rng.integers(0, 2, n)
y = (idx ^ (rng.random(n) < 0.1)).astype(int)
for vals in ([b"a", b"b"], [Decimal("1.5"), Decimal("2.5")], ["a", "b"]):
    df = pd.DataFrame({"d": pd.Series([vals[i] for i in idx], dtype=object), "target": y})
    m = Model(make_config("binary", n_estimators=5, n_splits=2))
    name = type(vals[0]).__name__
    try:
        m.fit(data=df)
    except Exception as exc:  # noqa: BLE001
        print(name, "fit", type(exc).__name__, str(exc)[:150])
        continue
    b = m._refit_result.model._model
    print(name, "fit ok; categorical_features", m._fit_result.categorical_features,
          "pandas_categorical", b.pandas_categorical)
    try:
        m.export(Path(tempfile.mkdtemp()) / "e")
        print(name, "export ok")
    except Exception as exc:  # noqa: BLE001
        print(name, "export", type(exc).__name__, str(exc)[-60:])
