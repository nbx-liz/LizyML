"""Write one small LightGBM model's text, to diff it across lightgbm versions."""

import sys

import lightgbm as lgb
import numpy as np

rng = np.random.default_rng(0)
X = rng.normal(size=(200, 3))
y = (X[:, 0] + rng.normal(scale=0.5, size=200) > 0).astype(int)
booster = lgb.train(
    {"objective": "binary", "verbose": -1, "seed": 0, "num_threads": 1},
    lgb.Dataset(X, y),
    num_boost_round=3,
)
with open(sys.argv[1], "w") as fh:
    fh.write(booster.model_to_string())
