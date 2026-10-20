"""H-0133: a raw-categories v2 booster given a frame of float codes (the codes alternative)."""

import lightgbm as lgb, numpy as np, pandas as pd
rng = np.random.default_rng(0); n = 300
c = pd.Categorical(rng.choice(["a", "b", "c"], n), categories=["a", "b", "c"])
x = rng.normal(size=n); y = (np.asarray(c.codes) + x > 1).astype(int)
df = pd.DataFrame({"c": c, "x": x})
b = lgb.train({"objective": "binary", "verbose": -1}, lgb.Dataset(df, y, categorical_feature=["c"]), 10)
codes = pd.DataFrame({"c": np.asarray(c.codes, dtype=float), "x": x})
try:
    out = b.predict(codes)
    print("codes frame: no error; equal =", bool(np.allclose(out, b.predict(df))))
except Exception as e:
    print("codes frame:", type(e).__name__, e)
