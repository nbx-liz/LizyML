"""#391: where does a fitted model still hold v (a frozenset with np.timedelta64)?

Fits ``[safe, safe, v, "x"]`` on the given tree and walks the model object
graph (attributes, dataclass fields, mappings, sequences, sets, numpy object
arrays, pandas objects), printing the path of every frozenset that holds an
``np.timedelta64``.

    PYTHONPATH=<tree> python -B docs/audits/2026-10-i391/probes/find_v.py
"""

from __future__ import annotations

import datetime as dt
import warnings
from typing import Any

import numpy as np
import pandas as pd

from lizyml import Model
from tests._helpers import make_config

warnings.simplefilter("ignore")
V = frozenset({np.uint64(4), np.timedelta64(4, "s")})
SAFE = frozenset({4, dt.timedelta(seconds=4)})
rng = np.random.default_rng(0)
rows = [SAFE, SAFE, V, "x"] * 20
df = pd.DataFrame(
    {
        "c": pd.Series(rows, dtype=object),
        "num": rng.normal(size=len(rows)),
        "target": rng.normal(size=len(rows)),
    }
)
model = Model(make_config("regression", n_estimators=5, n_splits=2))
model.fit(data=df)

seen: set[int] = set()
hits: list[str] = []


def walk(obj: Any, path: str, depth: int = 0) -> None:
    if depth > 12 or id(obj) in seen:
        return
    seen.add(id(obj))
    if type(obj) is frozenset:
        if any(type(m) is np.timedelta64 for m in obj):
            hits.append(path)
        return
    if isinstance(obj, (str, bytes, int, float, bool, type(None), np.generic)):
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            walk(v, f"{path}[{k!r}]", depth + 1)
        return
    if isinstance(obj, (list, tuple, set)):
        for i, v in enumerate(obj):
            walk(v, f"{path}[{i}]", depth + 1)
        return
    if isinstance(obj, np.ndarray):
        if obj.dtype == object:
            for i, v in enumerate(obj.ravel()[:5000]):
                walk(v, f"{path}.flat[{i}]", depth + 1)
        return
    if isinstance(obj, (pd.Series, pd.Index)):
        walk(obj.to_numpy(dtype=object), f"{path}.values", depth + 1)
        if isinstance(obj.dtype, pd.CategoricalDtype):
            walk(obj.cat.categories.to_numpy(dtype=object), f"{path}.categories", depth + 1)
        return
    if isinstance(obj, pd.DataFrame):
        for col in obj.columns:
            walk(obj[col], f"{path}[{col!r}]", depth + 1)
        return
    attrs = getattr(obj, "__dict__", None)
    if attrs is None and hasattr(obj, "__slots__"):
        attrs = {s: getattr(obj, s) for s in obj.__slots__ if hasattr(obj, s)}
    if isinstance(attrs, dict):
        for k, v in attrs.items():
            walk(v, f"{path}.{k}", depth + 1)


walk(model, "model")
print(f"{len(hits)} paths hold v:")
for p in sorted(set(hits))[:40]:
    print(" ", p)
