"""#423 firing-rate census: how often a default-space trial exceeds a memory budget.

Runs ``Model.tune`` with the default search space on the three vendored tutorial
datasets, replacing the per-trial ``check_resources`` with a probe that records
the real H-0140 assessment (finalized params, feature bins, booster count) and
then fails the trial before any training. With no COMPLETE trial TPE samples
at random, so the population is random draws from the default space's configured
distributions (learning_rate log-uniform, the others uniform).

    python budget_census.py <notebooks/data dir> <n_trials> <out.json> [--as-read]

``--as-read`` passes the frames as ``pandas.read_csv`` returns them (object
columns), as an upload reaches ``Model.tune``; without it the categorical
columns are cast to ``category`` first.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from typing import Any

import pandas as pd

import lizyml.core._model_tuning as mt
from lizyml import Model
from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.estimators.lgbm.memory_check import assess, feature_bins, finalized

RECORDS: list[dict[str, Any]] = []


def probe(provider: Any, tc: Any, X: pd.DataFrame, smart_params: dict[str, Any], *,
          n_boosters: int, stacklevel: int, latch: Any = None) -> None:
    params, rounds = finalized(tc.estimator_factory())
    a = assess(params, rounds, bins=feature_bins(X, params),
               auto_num_leaves=bool(smart_params.get("auto_num_leaves", False)),
               n_boosters=n_boosters, n_features=X.shape[1])
    RECORDS[-1]["trials"].append(
        {"bytes": a.estimate_bytes, "num_leaves": a.num_leaves, "max_depth": a.max_depth,
         "rounds": rounds, "n_boosters": n_boosters})
    raise LizyMLError(code=ErrorCode.TUNING_FAILED, user_message="census probe: not trained")


def datasets(root: Path, cast: bool) -> list[tuple[str, str, pd.DataFrame, str]]:
    credit = pd.read_csv(root / "credit-g.csv.gz")
    vehicle = pd.read_csv(root / "vehicle.csv.gz")
    target = vehicle.pop("Class")
    vehicle["vehicle_class"] = target.map({c: i for i, c in enumerate(sorted(target.unique()))})
    diamonds = pd.read_csv(root / "diamonds.csv.gz")
    if cast:  # LizyML's DataFrameBuilder casts object/string columns itself
        for col in ("cut", "color", "clarity"):
            diamonds[col] = diamonds[col].astype("category")
        for col in credit.columns:
            if credit[col].dtype == object and col != "class":
                credit[col] = credit[col].astype("category")
    return [("credit-g", "binary", credit, "class"),
            ("vehicle", "multiclass", vehicle, "vehicle_class"),
            ("diamonds", "regression", diamonds, "price")]


def main() -> None:
    root, n_trials, out = Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
    cast = sys.argv[4:] != ["--as-read"]
    mt.check_resources = probe  # the tune loop looks it up on the module
    for name, task, df, target in datasets(root, cast):
        RECORDS.append({"dataset": name, "task": task, "rows": len(df),
                        "features": df.shape[1] - 1, "trials": []})
        cfg = {"config_version": 1, "task": task, "data": {"target": target},
               "model": {"name": "lgbm", "auto_num_leaves": True},
               "split": {"method": "kfold" if task == "regression" else "stratified_kfold",
                         "n_splits": 5},
               "training": {"seed": 42},
               "tuning": {"optuna": {"params": {"n_trials": n_trials}}}}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                Model(cfg).tune(data=df)
            except LizyMLError as exc:
                if exc.code != ErrorCode.TUNING_FAILED:
                    raise
    out.write_text(json.dumps(RECORDS))
    for r in RECORDS:
        b = sorted(t["bytes"] for t in r["trials"])
        n = len(b)
        print(f"{r['dataset']} ({r['task']}, {r['rows']} rows, {r['features']} features): "
              f"{n} trials; max {b[-1] / 2**30:.2f} GiB; "
              + "; ".join(f"> {g} GiB: {sum(x > g * 2**30 for x in b)}/{n}" for g in (1, 2, 4, 8)))


if __name__ == "__main__":
    main()
