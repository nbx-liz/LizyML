"""Probe: two facts H-0148 states about today's behaviour (#409).

1. A frame whose duplicate column names are all in ``features.exclude`` fits.
2. ``Model.tune`` with a storage on a refused name writes a study and failed
   trials before ending with ``TUNING_FAILED``.
"""

from __future__ import annotations

import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

from lizyml import Model
from lizyml.core.exceptions import LizyMLError

BASE = {
    "config_version": 1,
    "task": "binary",
    "data": {"target": "target"},
    "split": {"method": "kfold", "n_splits": 3},
    "model": {"name": "lgbm", "params": {"n_estimators": 5}},
}


def main() -> int:
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(120, 3)), columns=["x", "a", "a"])
    df["target"] = (rng.normal(size=120) > 0).astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            Model({**BASE, "features": {"exclude": ["a"]}}).fit(data=df)
            print("excluded duplicate\tfit=ok")
        except Exception as exc:  # noqa: BLE001 - the probe reports every failure
            print(f"excluded duplicate\tfit={type(exc).__name__}: {exc}")

        bad = pd.DataFrame({"x": rng.normal(size=120), "a,b": rng.normal(size=120)})
        bad["target"] = (rng.normal(size=120) > 0).astype(int)
        cfg = {**BASE, "tuning": {"optuna": {"params": {"n_trials": 2}, "space": {}}}}
        with tempfile.TemporaryDirectory() as d:
            storage = f"sqlite:///{Path(d) / 'optuna.db'}"
            try:
                Model(cfg).tune(data=bad, storage=storage, study_name="s")
                print("tune with storage\tok")
            except LizyMLError as exc:
                print(f"tune with storage\t{exc.code.name}")
            summaries = optuna.study.get_all_study_summaries(storage)
            for s in summaries:
                study = optuna.load_study(study_name=s.study_name, storage=storage)
                states = [t.state.name for t in study.trials]
                print(f"stored study {s.study_name!r}\ttrials={states}")
            if not summaries:
                print("stored studies\tnone")
    return 0


if __name__ == "__main__":
    sys.exit(main())
