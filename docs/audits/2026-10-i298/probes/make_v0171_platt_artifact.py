"""Write the v0.17.1 platt artifact that #298 row 9a-9b loads (test fixture).

Run ONLY with LizyML 0.17.1 installed, the last release before H-0100. Use
the dependency floor (Python 3.10, numpy 1.24.0, pandas 2.0.0, lightgbm 4.4.0,
scipy 1.10.0, scikit-learn 1.3.0), so that every CI lane can read the pickles:
a numpy 2 pickle names ``numpy._core``, which numpy 1.24 lacks.

    VIRTUAL_ENV=<venv> uv pip install "lizyml[calibration]==0.17.1" -c floor-pins.txt
    <venv>/bin/python docs/audits/2026-10-i298/probes/make_v0171_platt_artifact.py <out_dir>

Output:
- ``<out_dir>/artifact/``: what ``Model.export`` wrote at v0.17.1;
- ``<out_dir>/expected.json``: the probe frame and what v0.17.1 predicted for
  it (``pred`` and ``proba``), plus the platt coefficients.

The script refuses to run on any other LizyML version.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm
import numpy as np
import pandas as pd
import sklearn

import lizyml
from lizyml import Model

if lizyml.__version__ != "0.17.1":
    raise SystemExit(f"needs lizyml 0.17.1, found {lizyml.__version__}")
if np.__version__ != "1.24.0":
    raise SystemExit(f"needs numpy 1.24.0 (the floor), found {np.__version__}")

out = Path(sys.argv[1])
rng = np.random.default_rng(298)
n = 300
X = pd.DataFrame({f"x{i}": rng.normal(size=n) for i in range(4)})
logit = 1.5 * X["x0"] - X["x1"] + 0.5 * rng.normal(size=n)
df = X.assign(target=(logit > 0).astype(int))

cfg = {
    "config_version": 1,
    "task": "binary",
    "data": {"target": "target"},
    "split": {"method": "stratified_kfold", "n_splits": 3, "random_state": 42},
    # A large learning rate so 100 rounds spread the scores (the default
    # learned too little on 300 rows to give the calibrator a range to map).
    "model": {"name": "lgbm", "params": {"n_estimators": 100, "learning_rate": 0.3}},
    "training": {"seed": 0},
    "calibration": {"method": "platt"},
}
model = Model(cfg, data=df)
model.fit()
artifact = Path(model.export(out / "artifact"))

# Rows from the training distribution (so the scores spread over the
# calibrator's range) and a few far outside it.
probe = pd.concat(
    [
        X.iloc[:30],
        pd.DataFrame({"x0": [-6.0, 6.0], "x1": [6.0, -6.0], "x2": [0.0, 0.0], "x3": [0.0, 0.0]}),
    ],
    ignore_index=True,
)
pred = model.predict(probe)
c_final = model._fit_result.calibrator.c_final  # type: ignore[union-attr]
lr = c_final._model
# The model's calibrated probabilities spread little on 300 rows (the OOF
# scores the calibrator learned from are weak), so the calibrator itself is
# also probed over a wide score range: that is what a wrong migration of the
# legacy LogisticRegression state would change.
# The score range is set from the learned slope so that the logit spans
# about [-4, 4]; the OOF raw scores are large here, so the slope is small.
half_width = 4.0 / max(abs(float(lr.coef_[0, 0])), 1e-12)
scores = np.linspace(-half_width, half_width, 33)
calibrated = np.asarray(c_final.predict(scores))
if float(np.ptp(calibrated)) < 0.5:
    raise SystemExit(f"calibrator probe does not spread (coef {lr.coef_}, intercept {lr.intercept_}, range {calibrated.min()}..{calibrated.max()})")
expected = {
    "lizyml": lizyml.__version__,
    "numpy": np.__version__,
    "pandas": pd.__version__,
    "lightgbm": lightgbm.__version__,
    "scikit-learn": sklearn.__version__,
    "probe": probe.to_dict(orient="list"),
    "pred": np.asarray(pred.pred).tolist(),
    "proba": np.asarray(pred.proba).tolist(),
    "platt_coef": float(lr.coef_[0, 0]),
    "platt_intercept": float(lr.intercept_[0]),
    "calibrator_scores": scores.tolist(),
    "calibrator_proba": calibrated.tolist(),
}
(out / "expected.json").write_text(json.dumps(expected, indent=1), encoding="utf-8")
print("wrote", artifact, "files:", sorted(p.name for p in artifact.iterdir()))
print("sizes:", {p.name: p.stat().st_size for p in artifact.iterdir()})
