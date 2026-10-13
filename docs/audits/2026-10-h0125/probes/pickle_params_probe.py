"""Does a joblib round trip keep lgb.Booster.params? (H-0125 criterion 8 note)."""

import io

import joblib
import lightgbm as lgb
import numpy as np

X = np.random.default_rng(0).normal(size=(200, 3))
y = (X[:, 0] > 0).astype(float)
b = lgb.train({"objective": "cross_entropy_lambda", "verbose": -1}, lgb.Dataset(X, y), 3)
buf = io.BytesIO()
joblib.dump(b, buf)
buf.seek(0)
b2 = joblib.load(buf)
print("lightgbm", lgb.__version__)
print("pickled params:", b2.params.get("objective") if b2.params else b2.params)
print("model_str params:", lgb.Booster(model_str=b.model_to_string()).params)
