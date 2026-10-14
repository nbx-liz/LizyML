"""Platt TNC case under numpy 1.x vs 2.x: does the input data differ, or the fit?"""

import hashlib

import numpy as np
import pandas as pd

from lizyml.calibration.platt import PlattCalibrator
from lizyml.core.model import Model
from tests._helpers import make_binary_df, make_config

print("numpy", np.__version__, "pandas", pd.__version__)
df = make_binary_df(n=240, seed=6)
h = hashlib.sha256(pd.util.hash_pandas_object(df, index=True).to_numpy().tobytes())
print("data sha256", h.hexdigest()[:16], "dtypes", dict(df.dtypes.astype(str)))

cfg = make_config("binary", n_estimators=5, n_splits=3)
model = Model(cfg, data=df)
fr = model.fit()
oof = np.asarray(fr.oof_pred, dtype=np.float64).ravel()
print("oof sha256", hashlib.sha256(oof.tobytes()).hexdigest()[:16], "dtype", fr.oof_pred.dtype)
y = df["target"].to_numpy()
logit = np.log(oof / (1 - oof))
for name, scores in (("logit(oof)", logit), ("oof", oof)):
    cal = PlattCalibrator({"target_smoothing": False, "method": "TNC"}) if False else None
    try:
        from lizyml.calibration.registry import get_calibrator

        c = get_calibrator("platt", {"target_smoothing": False, "method": "TNC"})
        c.fit(scores, y)
        print(name, "fit ok", c._coef)
    except Exception as e:  # noqa: BLE001 - the probe reports what escapes
        print(name, type(e).__name__, str(e)[:90])
