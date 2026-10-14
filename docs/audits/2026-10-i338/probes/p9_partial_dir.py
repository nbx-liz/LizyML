"""#338: what a failed ``Model.export`` leaves in the output directory today."""

from __future__ import annotations

import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from lizyml import Model
from tests._helpers import make_config

warnings.filterwarnings("ignore")
rng = np.random.default_rng(0)
n = 120
values = [b"a", b"b", b"c"]
d = pd.Categorical([values[i] for i in rng.integers(0, 3, n)], categories=values)
x = rng.normal(size=n)
df = pd.DataFrame({"d": d, "x": x, "target": (x > 0).astype(int)})
model = Model(make_config("binary"), data=df)
model.fit()

with tempfile.TemporaryDirectory() as tmp:
    out = Path(tmp) / "export"
    try:
        model.export(out)
        print("export: ok")
    except Exception as exc:  # noqa: BLE001 - the probe reports what escapes
        print(f"export: {type(exc).__name__}: {str(exc).replace(tmp, '<tmp>')}")
    print("directory exists:", out.exists())
    if out.exists():
        for p in sorted(out.iterdir()):
            print(f"  {p.name} {p.stat().st_size} bytes")
