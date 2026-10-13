"""Probe: where a LightGBM booster records the objective it was trained with.

H-0125 refuses at load an artifact whose boosters were trained with
``cross_entropy_lambda``. This shows which record survives a model-string
round trip (``lgb.Booster(model_str=...)``): ``params`` does not, the model
string's ``objective=`` line and ``dump_model()["objective"]`` do. A joblib
(pickle) round trip, the form LizyML stores, keeps ``params`` on LightGBM
4.6.0 (H-0125 実装の記録); the load check reads the recorded objective either
way and never ``params``.
"""

import lightgbm as lgb
import numpy as np

X = np.random.default_rng(0).normal(size=(200, 3))
y = (X[:, 0] > 0).astype(float)
for obj in ("cross_entropy_lambda", "xentlambda", "binary", "cross_entropy"):
    b = lgb.train({"objective": obj, "verbose": -1}, lgb.Dataset(X, y), num_boost_round=3)
    text = [ln for ln in b.model_to_string().splitlines() if ln.startswith("objective=")]
    print(obj, "| dump:", b.dump_model()["objective"], "| text:", text, "| params:", b.params.get("objective"))
    b2 = lgb.Booster(model_str=b.model_to_string())
    print("   reloaded | dump:", b2.dump_model()["objective"], "| params:", b2.params.get("objective"))
