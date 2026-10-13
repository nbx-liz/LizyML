"""Probe: native LightGBM huber metric vs LizyML HuberLoss on the same predictions."""

import lightgbm as lgb
import numpy as np

from lizyml.metrics.regression import HuberLoss

rng = np.random.default_rng(0)
X = rng.normal(size=(500, 3))
y = 3 * X[:, 0] + rng.normal(scale=2.0, size=500)
ds = lgb.Dataset(X, y)
for metric_params in ({}, {"alpha": 9.0}):
    rec: dict = {}
    params = {"objective": "regression", "metric": "huber", "verbose": -1, **metric_params}
    booster = lgb.train(
        params,
        ds,
        num_boost_round=20,
        valid_sets=[ds],
        callbacks=[lgb.record_evaluation(rec)],
    )
    pred = booster.predict(X)
    native = rec["training"]["huber"][-1]
    print(
        metric_params,
        "native huber:",
        round(native, 6),
        "LizyML delta=1.0:",
        round(HuberLoss()(y, pred), 6),
        "LizyML delta=0.9:",
        round(HuberLoss(delta=0.9)(y, pred), 6),
        "LizyML delta=9.0:",
        round(HuberLoss(delta=9.0)(y, pred), 6),
    )
