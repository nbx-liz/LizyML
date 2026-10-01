"""Probe: does an environment override reach config_version after the loader check?"""

import os
import warnings

from lizyml.config.loader import load_config
from lizyml.core.exceptions import LizyMLError

warnings.filterwarnings("ignore")
BASE = {
    "config_version": 1, "task": "binary", "data": {"target": "y"},
    "split": {"method": "kfold", "n_splits": 2},
    "model": {"name": "lgbm"}, "evaluation": {"metrics": ["auc"]},
}
for env_value in ["2", "1", "0", "false"]:
    os.environ["LIZYML__config_version"] = env_value
    try:
        cfg = load_config(dict(BASE))
        print(f"env LIZYML__config_version={env_value!r}: accepted, config_version={cfg.config_version!r}")
    except LizyMLError as e:
        print(f"env LIZYML__config_version={env_value!r}: LizyMLError {e.code.value}")
