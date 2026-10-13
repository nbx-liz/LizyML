"""#348 and #349 reproduced on the current tree (the issues' own cases).

#348: a resumed round whose trials all fail closes ``completed`` and tune()
returns round 1's best. #349: a LightGBM error leaves the round open (the issues used `num_leaves`,
which H-0121 since made smart-managed; `max_depth` takes its place); finishing
it without ``space=`` fails again, and a corrected ``space=`` is refused.
"""

from __future__ import annotations

import tempfile
import warnings
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

from lizyml import Model
from lizyml.core.exceptions import LizyMLError
from lizyml.tuning.study_lineage import read_study_lineage
from tests._helpers import make_config

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.ERROR)

rng = np.random.default_rng(0)
n = 120
df = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
df["target"] = df["a"] * 2 + rng.normal(scale=0.1, size=n)
cfg = make_config("regression", n_estimators=3, n_splits=2, tuning_n_trials=2)

GOOD = {
    "max_depth": {"type": "int", "low": 3, "high": 6},
    "learning_rate": {"type": "float", "low": 0.05, "high": 0.3, "log": True},
}


def states(storage: str, name: str) -> str:
    study = optuna.load_study(study_name=name, storage=storage)
    return ", ".join(t.state.name for t in study.trials)


def lineage_line(storage: str, name: str) -> str:
    lineage = read_study_lineage(storage=storage, study_name=name)
    rounds = [(r.round, r.status) for r in lineage.rounds]
    return f"state={lineage.state} rounds={rounds}"


def call(label: str, fn) -> None:  # noqa: ANN001
    try:
        result = fn()
        rounds = [(r.round, r.n_trials) for r in result.rounds]
        print(f"{label}: returned; rounds={rounds} best_score={result.best_score:.6f}")
    except LizyMLError as exc:
        print(f"{label}: {exc.code.name}: {exc.user_message[:150]}")


with tempfile.TemporaryDirectory() as tmp:
    storage = f"sqlite:///{Path(tmp) / 's.db'}"
    kw = dict(storage=storage, expand_boundary=False)

    print("== #348")
    call("round 1", lambda: Model(cfg, data=df).tune(study_name="s348", **kw))
    bad_ratio = {
        **GOOD,
        "validation_ratio": {"type": "float", "low": 1.0, "high": 1.5, "category": "training"},
    }
    call(
        "round 2 (every trial fails)",
        lambda: Model(cfg, data=df).tune(resume=True, study_name="s348", space=bad_ratio, **kw),
    )
    print("trials:", states(storage, "s348"))
    print("lineage:", lineage_line(storage, "s348"))

    print("== #349")
    call("round 1", lambda: Model(cfg, data=df).tune(study_name="s349", **kw))
    bad_bin = {**GOOD, "max_bin": {"type": "int", "low": 0, "high": 1}}
    call(
        "1: space with max_bin 0-1",
        lambda: Model(cfg, data=df).tune(resume=True, study_name="s349", space=bad_bin, **kw),
    )
    call(
        "2: finish without space=",
        lambda: Model(cfg, data=df).tune(resume=True, study_name="s349", **kw),
    )
    call(
        "3: finish with a corrected space=",
        lambda: Model(cfg, data=df).tune(resume=True, study_name="s349", space=GOOD, **kw),
    )
    print("trials:", states(storage, "s349"))
    print("lineage:", lineage_line(storage, "s349"))
