"""Is a booster trained with categorical names byte-identical to one trained with positions?"""

import hashlib

import lightgbm as lgb
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
n = 200
X = pd.DataFrame(
    {
        "num": rng.normal(size=n),
        "c": pd.Categorical(rng.choice(["a", "b", "c"], size=n)),
        "d": pd.Categorical(rng.choice(["p", "q"], size=n)),
    }
)
y = rng.normal(size=n)
params = {"objective": "regression", "verbose": -1, "num_threads": 1, "seed": 0}
out = {}
for name, cat in {"names": ["c", "d"], "positions": [1, 2], "auto": "auto"}.items():
    ds = lgb.Dataset(X, label=y, categorical_feature=cat, free_raw_data=False)
    booster = lgb.train(params, ds, num_boost_round=5)
    text = booster.model_to_string()
    out[name] = hashlib.sha256(text.encode()).hexdigest()
    print(name, out[name][:16], booster.params.get("categorical_column"))
    print("   categorical_feature lines:", [ln for ln in text.splitlines() if "categorical" in ln][:4])
print("names == positions:", out["names"] == out["positions"])
