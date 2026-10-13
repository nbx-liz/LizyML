"""Which calibration stage fails under numpy 1.x, and does the generated fitter
refuse the same scores?  Run from the repo root with PYTHONPATH=."""

import sys
import tempfile
from pathlib import Path

import numpy as np

from lizyml import Model
from lizyml.calibration.registry import get_calibrator
from lizyml.core.exceptions import LizyMLError
from tests._helpers import make_binary_df, make_config
from tests.test_codegen.test_calibration_params_codegen import _load_train

PARAMS = {"target_smoothing": False, "method": "TNC"}
print("numpy", np.__version__)
df = make_binary_df(n=240, seed=6)

cfg = make_config("binary", n_estimators=5, n_splits=3, calibration="platt", calibration_params=PARAMS)
try:
    Model(cfg, data=df).fit()
    print("runtime fit: ok")
except LizyMLError as e:
    print("runtime fit:", e.code, {k: v for k, v in e.context.items() if k in ("stage", "fold", "method")})

# Same model without the TNC params: OOF scores are unchanged by calibration.
cfg_ok = make_config("binary", n_estimators=5, n_splits=3, calibration="platt", calibration_params=None)
m = Model(cfg_ok, data=df)
fr = m.fit()
with tempfile.TemporaryDirectory() as d:
    out = Path(d) / "x"
    m.export_code(out)
    train = _load_train(out)
    oof = np.asarray(fr.oof_raw_scores, dtype=np.float64)
    scores = np.log(oof / (1 - oof)) if oof.min() > 0 and oof.max() < 1 else oof
    y = df["target"].to_numpy()
    for label, s in (("oof_raw_scores (c_final input)", oof),):
        r = g = None
        try:
            r = get_calibrator("platt", params=PARAMS).fit(s, y).export_params()
        except LizyMLError as e:
            r = f"refused {e.code}"
        try:
            g = train._CAL_FITTERS["platt"](s, y, PARAMS)
        except Exception as e:  # noqa: BLE001 - the probe reports what escapes
            g = f"refused {type(e).__name__}"
        print(f"{label}: runtime={r} | generated={g}")
sys.exit(0)
