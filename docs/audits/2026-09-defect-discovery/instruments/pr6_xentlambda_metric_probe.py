"""Probe: Model.fit with cross_entropy_lambda and only ranking/ECE metrics (data from #307)."""

import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, "tests")
from _helpers import make_config  # noqa: E402

from lizyml.core.model import Model  # noqa: E402

warnings.filterwarnings("ignore")
rng = np.random.default_rng(0)
X = rng.normal(size=(1500, 3))
df = pd.DataFrame(X, columns=["a", "b", "c"])
df["target"] = (2 * X[:, 0] + rng.normal(scale=0.5, size=1500) > 0).astype(int)
for metrics in (["auc"], ["auc_pr"], ["ece"], ["precision_at_k"], ["logloss"], ["brier"]):
    cfg = make_config("binary", n_estimators=300, n_splits=3)
    cfg["model"]["params"].update(objective="cross_entropy_lambda", learning_rate=0.1)
    cfg["evaluation"] = {"metrics": metrics}
    m = Model(cfg)
    try:
        m.fit(data=df)
        p = m.predict(df[["a", "b", "c"]]).proba
        print(metrics, "fit OK", m.evaluate()["raw"]["oof"], f"predict proba {p.min():.3f}..{p.max():.3f}")
    except Exception as e:  # noqa: BLE001
        print(metrics, "FAILS", type(e).__name__, str(e)[:90])
