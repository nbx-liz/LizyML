"""Probe: which "huber" each FitResult surface reports under default regression params."""

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "tests")
from _helpers import make_config  # noqa: E402

from lizyml import Model  # noqa: E402
from lizyml.metrics.regression import HuberLoss  # noqa: E402

rng = np.random.default_rng(0)
X = rng.normal(size=(600, 3))
df = pd.DataFrame(X, columns=["a", "b", "c"])
df["target"] = 3 * X[:, 0] + rng.normal(scale=2.0, size=600)
cfg = make_config("regression", n_estimators=30, n_splits=3)
cfg["evaluation"] = {"metrics": ["huber"]}
print("model.params:", cfg["model"].get("params"))
model = Model(cfg)
res = model.fit(data=df)
hist = res.history[0]
print("history[0] keys:", sorted(hist))
print("eval_history structure:", {k: sorted(v) if isinstance(v, dict) else type(v).__name__ for k, v in hist["eval_history"].items()})
print("FitResult.metrics raw:", {k: v for k, v in res.metrics["raw"].items() if k in ("oof", "oof_per_fold")})
y = df["target"].to_numpy()
oof = np.asarray(res.oof_pred)
print("OOF HuberLoss delta=1.0:", HuberLoss()(y, oof), " delta=0.9:", HuberLoss(delta=0.9)(y, oof))
