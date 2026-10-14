"""Probe: LightGBM native metric vs LizyML metric of the same name, same predictions."""

import lightgbm as lgb
import numpy as np

from lizyml.metrics.registry import get_metric

rng = np.random.default_rng(0)
X = rng.normal(size=(500, 3))
# targets with |y| < 1 rows, so a max(1, |y|) denominator shows
y = 0.8 * X[:, 0] + rng.normal(scale=0.5, size=500)
ds = lgb.Dataset(X, y)
rec: dict = {}
booster = lgb.train(
    {"objective": "regression", "metric": ["rmse", "l1", "mape", "huber"], "verbose": -1},
    ds,
    num_boost_round=20,
    valid_sets=[ds],
    callbacks=[lgb.record_evaluation(rec)],
)
pred = booster.predict(X)
for native, ours in (("rmse", "rmse"), ("l1", "mae"), ("mape", "mape"), ("huber", "huber")):
    print(f"{native:6s} native={rec['training'][native][-1]:.6f} lizyml {ours}={get_metric(ours)(y, pred):.6f}")

yb = (X[:, 0] + rng.normal(scale=0.5, size=500) > 0).astype(float)
dsb = lgb.Dataset(X, yb)
recb: dict = {}
bb = lgb.train(
    {"objective": "binary", "metric": ["binary_logloss", "auc", "average_precision"], "verbose": -1},
    dsb,
    num_boost_round=20,
    valid_sets=[dsb],
    callbacks=[lgb.record_evaluation(recb)],
)
p = bb.predict(X)
for native, ours in (("binary_logloss", "logloss"), ("auc", "auc"), ("average_precision", "auc_pr")):
    print(f"{native:18s} native={recb['training'][native][-1]:.6f} lizyml {ours}={get_metric(ours)(yb, p):.6f}")
