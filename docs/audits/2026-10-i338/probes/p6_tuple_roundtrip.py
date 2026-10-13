"""Q5 supplement: tuple categories through export/load; pure-Python json path."""

import json
import json.encoder
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from lizyml import Model
from tests._helpers import make_config

warnings.filterwarnings("ignore")

# Pure-Python encoder path gives the same message (C encoder disabled).
json.encoder.c_make_encoder = None
from lightgbm.basic import _json_default_with_numpy  # noqa: E402

try:
    json.dumps([[b"a"]], default=_json_default_with_numpy)
except ValueError as exc:
    print("pure-python encoder:", type(exc).__name__, exc)

vals = [(1, 2), (3, 4), (5, 6)]
rng = np.random.default_rng(0)
n = 150
idx = rng.integers(0, 3, n)
y = ((idx == 0) ^ (rng.random(n) < 0.1)).astype(int)
df = pd.DataFrame({"d": pd.Categorical([vals[i] for i in idx], categories=vals), "target": y})
m = Model(make_config("binary", n_estimators=5, n_splits=2))
m.fit(data=df)
p = m.predict(df[["d"]]).proba
tmp = Path(tempfile.mkdtemp()) / "e"
m.export(tmp)
ml = Model.load(tmp)
b = ml._refit_result.model._model
print("loaded pandas_categorical:", b.pandas_categorical)
print("model string tail:", b.model_to_string().strip().splitlines()[-1])
rev = pd.DataFrame({"d": pd.Categorical([vals[i] for i in idx], categories=vals[::-1])})
pl = ml.predict(rev).proba
print("max|p - p_loaded(reversed cat order)|:", float(np.max(np.abs(np.asarray(p) - np.asarray(pl)))))
print("proba range:", float(np.min(p)), float(np.max(p)))
