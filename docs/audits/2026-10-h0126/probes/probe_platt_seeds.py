"""Which make_binary_df seeds let platt + TNC (no smoothing) fit, on this numpy?"""

import warnings

import numpy as np

from lizyml import Model
from lizyml.core.exceptions import LizyMLError
from tests._helpers import make_binary_df, make_config

PARAMS = {"target_smoothing": False, "method": "TNC"}
warnings.simplefilter("ignore")
ok = []
for seed in range(12):
    cfg = make_config(
        "binary", n_estimators=5, n_splits=3, calibration="platt", calibration_params=PARAMS
    )
    try:
        Model(cfg, data=make_binary_df(n=240, seed=seed)).fit()
        ok.append(seed)
    except LizyMLError as e:
        print(f"seed {seed}: {e.code.value} {e.context.get('stage')}")
print("numpy", np.__version__, "ok seeds", ok)
