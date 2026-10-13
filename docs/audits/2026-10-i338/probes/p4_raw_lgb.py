"""Q6: raw LightGBM (no LizyML) with bytes categories, any installed version.

Also a Q5 negative control: a str-coded model with a *permuted* mapping must
differ, so the 0.0 differences in p2 are not an artefact of constant output.
"""

import pickle
from decimal import Decimal

import lightgbm as lgb
import numpy as np
import pandas as pd

print("lightgbm", lgb.__version__, "pandas", pd.__version__)
rng = np.random.default_rng(0)
n = 200
idx = rng.integers(0, 3, n)
y = ((idx == 0) ^ (rng.random(n) < 0.1)).astype(int)
params = {"objective": "binary", "verbose": -1, "min_data_in_leaf": 5, "seed": 0}

for vals in ([b"a", b"b", b"c"], [Decimal(1), Decimal(2), Decimal(3)], ["a", "b", "c"]):
    X = pd.DataFrame({"d": pd.Categorical([vals[i] for i in idx], categories=vals)})
    try:
        lgb.train(params, lgb.Dataset(X, label=y), num_boost_round=5)
        print(type(vals[0]).__name__, "train(keep_training_booster=False) ok")
    except Exception as exc:  # noqa: BLE001
        print(type(vals[0]).__name__, "train(keep_training_booster=False)", type(exc).__name__, exc)
    b = lgb.train(params, lgb.Dataset(X, label=y), num_boost_round=5, keep_training_booster=True)
    p = b.predict(X)
    for label, fn in (
        ("model_to_string", b.model_to_string),
        ("pickle.dumps", lambda: pickle.dumps(b)),
    ):
        try:
            fn()
            print(type(vals[0]).__name__, label, "ok")
        except Exception as exc:  # noqa: BLE001
            print(type(vals[0]).__name__, label, type(exc).__name__, exc)
    print(type(vals[0]).__name__, "pred range", round(p.min(), 4), round(p.max(), 4))

# Negative control: permuted category mapping changes predictions.
codes = idx
perm = np.array([2, 0, 1])
Xa = pd.DataFrame({"d": pd.Categorical.from_codes(codes, categories=["s0", "s1", "s2"])})
Xb = pd.DataFrame({"d": pd.Categorical.from_codes(perm[codes], categories=["s0", "s1", "s2"])})
pa = lgb.train(params, lgb.Dataset(Xa, label=y), num_boost_round=5).predict(Xa)
Xa_wrong = pd.DataFrame({"d": pd.Categorical.from_codes(perm[codes], categories=["s0", "s1", "s2"])})
ba = lgb.train(params, lgb.Dataset(Xa, label=y), num_boost_round=5)
print("neg control max|p(correct) - p(permuted input)|:", float(np.max(np.abs(ba.predict(Xa) - ba.predict(Xa_wrong)))))
