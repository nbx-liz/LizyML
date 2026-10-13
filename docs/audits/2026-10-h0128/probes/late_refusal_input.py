"""H-0128 criterion 7: which tune(space=) refusal happens after round planning.

For each candidate ``space=`` on a resumed (completed-round) study, record
whether it is refused and which LizyML function raised: ``parse_space_arg``
(before the study is read and the round is planned) or ``check_round_space``
(after ``plan_round``). Criterion 7 needs an input refused by the latter.
"""

from __future__ import annotations

import tempfile
import traceback
import warnings
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

from lizyml import Model
from lizyml.core.exceptions import LizyMLError
from tests._helpers import make_config

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.ERROR)

rng = np.random.default_rng(0)
n = 80
df = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
df["target"] = (df["a"] > 0).astype(int)
cfg = make_config("binary", n_estimators=3, n_splits=2, tuning_n_trials=1)

LR = {"type": "float", "low": 0.05, "high": 0.3, "log": True}
CANDIDATES = {
    "int bound above 2**52": {"learning_rate": LR, "max_depth": {"type": "int", "low": 3, "high": 2**53}},
    "binary objective cross_entropy_lambda": {
        "learning_rate": LR,
        "objective": {"type": "categorical", "choices": ["binary", "cross_entropy_lambda"]},
    },
    "smart-managed num_leaves": {"learning_rate": LR, "num_leaves": {"type": "int", "low": 4, "high": 8}},
}

with tempfile.TemporaryDirectory() as tmp:
    storage = f"sqlite:///{Path(tmp) / 's.db'}"
    Model(cfg, data=df).tune(storage=storage, study_name="s", expand_boundary=False)
    for label, space in CANDIDATES.items():
        try:
            Model(cfg, data=df).tune(
                resume=True, storage=storage, study_name="s", expand_boundary=False, space=space
            )
            print(f"{label}: accepted")
        except LizyMLError as exc:
            names = [f.name for f in traceback.extract_tb(exc.__traceback__)]
            where = [n for n in ("parse_space_arg", "plan_round", "check_round_space") if n in names]
            frames = [f for f in traceback.extract_tb(exc.__traceback__) if "/lizyml/" in f.filename]
            inner = f"{frames[-1].name} ({Path(frames[-1].filename).name}:{frames[-1].lineno})" if frames else "?"
            print(f"{label}: {exc.code.name}; raised through {where}; innermost {inner}")
