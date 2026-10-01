"""#268: executed reachability of each census knob.

For every knob the static scan (``pr8_construction_sites.py``) shows bound from
Config or a public argument, set a NON-default value there, run the cheapest
real ``Model`` call that constructs the class, and record what the
constructor actually received (``__init__`` spied with ``inspect.signature``).

Prints one line per (scenario, knob): the value set, the value received, and
OK / MISMATCH / NOT-CONSTRUCTED. A knob is "reachable from Config" only on an
OK line; the static scan is not evidence on its own.
"""

from __future__ import annotations

import inspect
import sys
import warnings
from collections import defaultdict
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

from lizyml.calibration.beta import BetaCalibrator  # noqa: E402
from lizyml.calibration.isotonic import IsotonicCalibrator  # noqa: E402
from lizyml.calibration.platt import PlattCalibrator  # noqa: E402
from lizyml.core.model import Model  # noqa: E402
from lizyml.estimators.lgbm.adapter import LGBMAdapter  # noqa: E402
from lizyml.evaluation.evaluator import Evaluator  # noqa: E402
from lizyml.features.encoders.categorical_encoder import CategoricalEncoder  # noqa: E402
from lizyml.features.pipelines_native import NativeFeaturePipeline  # noqa: E402
from lizyml.metrics.classification import ECE, PrecisionAtK  # noqa: E402
from lizyml.metrics.regression import HuberLoss  # noqa: E402
from lizyml.splitters.blocked_group_kfold import BlockedGroupKFoldSplitter  # noqa: E402
from lizyml.splitters.group_kfold import (  # noqa: E402
    GroupKFoldSplitter,
    StratifiedGroupKFoldSplitter,
)
from lizyml.splitters.group_time_series import GroupTimeSeriesSplitter  # noqa: E402
from lizyml.splitters.kfold import KFoldSplitter, StratifiedKFoldSplitter  # noqa: E402
from lizyml.splitters.purged_time_series import PurgedTimeSeriesSplitter  # noqa: E402
from lizyml.splitters.time_series import TimeSeriesSplitter  # noqa: E402
from lizyml.training.cv_trainer import CVTrainer  # noqa: E402
from lizyml.training.inner_valid import (  # noqa: E402
    GroupHoldoutInnerValid,
    HoldoutInnerValid,
    TimeHoldoutInnerValid,
)
from lizyml.training.refit_trainer import RefitTrainer  # noqa: E402
from lizyml.tuning.tuner import Tuner  # noqa: E402

SPIED = [
    BetaCalibrator, IsotonicCalibrator, PlattCalibrator, LGBMAdapter, Evaluator,
    CategoricalEncoder, NativeFeaturePipeline, ECE, PrecisionAtK, HuberLoss,
    BlockedGroupKFoldSplitter, GroupKFoldSplitter, StratifiedGroupKFoldSplitter,
    GroupTimeSeriesSplitter, KFoldSplitter, StratifiedKFoldSplitter,
    PurgedTimeSeriesSplitter, TimeSeriesSplitter, CVTrainer, HoldoutInnerValid,
    GroupHoldoutInnerValid, TimeHoldoutInnerValid, RefitTrainer, Tuner,
]
received: dict[str, list[dict[str, Any]]] = defaultdict(list)


def _spy(cls: type) -> None:
    real = cls.__init__
    sig = inspect.signature(real)

    def init(self: Any, *args: Any, **kwargs: Any) -> None:
        bound = sig.bind(self, *args, **kwargs)
        bound.apply_defaults()
        received[cls.__name__].append(dict(list(bound.arguments.items())[1:]))
        real(self, *args, **kwargs)

    cls.__init__ = init  # type: ignore[method-assign]


for _c in SPIED:
    _spy(_c)

N = 240
rng = np.random.default_rng(0)


def frame(task: str) -> pd.DataFrame:
    df = pd.DataFrame({"a": rng.normal(size=N), "b": rng.normal(size=N)})
    df["g"] = np.repeat(np.arange(24), N // 24)
    df["t"] = np.arange(N)
    df["blk"] = np.repeat(np.arange(4), N // 4)
    signal = df["a"] + rng.normal(scale=0.5, size=N)
    df["y"] = signal if task == "regression" else (signal > 0).astype(int)
    return df


def config(task: str, split: dict[str, Any], **extra: Any) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "config_version": 1, "task": task,
        "data": {"target": "y"},
        "features": {"exclude": ["g", "t", "blk"]},
        "split": split,
        "model": {"name": "lgbm", "params": {"n_estimators": 5}},
    }
    for key, value in extra.items():
        cfg[key] = value
    return cfg


def run(label: str, expected: dict[str, Any], action: Callable[[], object]) -> None:
    received.clear()
    try:
        action()
        status = ""
    except Exception as exc:  # noqa: BLE001 -- the outcome is the measurement
        status = f"  (call raised {type(exc).__name__}: {str(exc)[:60]})"
    for knob, want in expected.items():
        cls, param = knob.split(".")
        calls = received.get(cls, [])
        if not calls:
            print(f"{label:22s} {knob:44s} set={want!r:12} NOT-CONSTRUCTED{status}")
            continue
        got = sorted({repr(c.get(param)) for c in calls})
        verdict = "OK" if got == [repr(want)] else "MISMATCH"
        print(f"{label:22s} {knob:44s} set={want!r:12} got={','.join(got):24s} {verdict}{status}")


def fit(task: str, split: dict[str, Any], **extra: Any) -> Callable[[], object]:
    def go() -> object:
        df = frame(task)
        cfg = config(task, split, **extra)
        if split["method"] in ("group_kfold", "stratified_group_kfold", "group_time_series"):
            cfg["data"]["group_col"] = "g"
        if split["method"] in ("time_series", "purged_time_series", "group_time_series"):
            cfg["data"]["time_col"] = "t"
        return Model(cfg).fit(data=df)

    return go


def main() -> None:
    run("kfold", {"KFoldSplitter.n_splits": 3, "KFoldSplitter.shuffle": False,
                  "KFoldSplitter.random_state": 7},
        fit("regression", {"method": "kfold", "n_splits": 3, "shuffle": False, "random_state": 7}))
    run("stratified_kfold", {"StratifiedKFoldSplitter.n_splits": 3,
                             "StratifiedKFoldSplitter.random_state": 7},
        fit("binary", {"method": "stratified_kfold", "n_splits": 3, "random_state": 7}))
    run("group_kfold", {"GroupKFoldSplitter.n_splits": 3},
        fit("regression", {"method": "group_kfold", "n_splits": 3}))
    run("strat_group_kfold", {"StratifiedGroupKFoldSplitter.n_splits": 3,
                              "StratifiedGroupKFoldSplitter.shuffle": False,
                              "StratifiedGroupKFoldSplitter.random_state": 7},
        fit("binary", {"method": "stratified_group_kfold", "n_splits": 3, "shuffle": False,
                       "random_state": 7}))
    run("time_series", {"TimeSeriesSplitter.n_splits": 3, "TimeSeriesSplitter.gap": 2,
                        "TimeSeriesSplitter.max_train_size": 100,
                        "TimeSeriesSplitter.max_test_size": 30},
        fit("regression", {"method": "time_series", "n_splits": 3, "gap": 2,
                           "train_size_max": 100, "test_size_max": 30}))
    run("purged_time_series", {"PurgedTimeSeriesSplitter.n_splits": 3,
                               "PurgedTimeSeriesSplitter.purge_gap": 2,
                               "PurgedTimeSeriesSplitter.embargo": 1,
                               "PurgedTimeSeriesSplitter.max_train_size": 100,
                               "PurgedTimeSeriesSplitter.max_test_size": 30},
        fit("regression", {"method": "purged_time_series", "n_splits": 3, "purge_gap": 2,
                           "embargo": 1, "train_size_max": 100, "test_size_max": 30}))
    run("group_time_series", {"GroupTimeSeriesSplitter.n_splits": 3,
                              "GroupTimeSeriesSplitter.gap": 1,
                              "GroupTimeSeriesSplitter.max_train_size": 12,
                              "GroupTimeSeriesSplitter.max_test_size": 4},
        fit("regression", {"method": "group_time_series", "n_splits": 3, "gap": 1,
                           "train_size_max": 12, "test_size_max": 4}))
    run("blocked_group_kfold", {"BlockedGroupKFoldSplitter.mode": "sliding",
                                "BlockedGroupKFoldSplitter.train_window": 2,
                                "BlockedGroupKFoldSplitter.n_splits": 2,
                                "BlockedGroupKFoldSplitter.shuffle": False,
                                "BlockedGroupKFoldSplitter.min_train_rows": 3,
                                "BlockedGroupKFoldSplitter.min_valid_rows": 2},
        fit("regression", {"method": "blocked_group_kfold",
                           "blocks": {"col": "blk", "cutoffs": [1, 2], "mode": "sliding",
                                      "train_window": 2},
                           "groups": {"col": "g", "n_splits": 2, "shuffle": False},
                           "min_train_rows": 3, "min_valid_rows": 2}))
    es = {"enabled": True, "rounds": 7,
          "inner_valid": {"method": "holdout", "ratio": 0.2, "random_state": 7, "stratify": True}}
    run("inner holdout + lgbm", {"HoldoutInnerValid.ratio": 0.2,
                                 "HoldoutInnerValid.random_state": 7,
                                 "HoldoutInnerValid.stratify": True,
                                 "LGBMAdapter.early_stopping_rounds": 7,
                                 "LGBMAdapter.random_state": 7},
        fit("binary", {"method": "kfold", "n_splits": 3},
            training={"seed": 7, "early_stopping": es}))
    run("inner group_holdout", {"GroupHoldoutInnerValid.ratio": 0.2,
                                "GroupHoldoutInnerValid.random_state": 7},
        fit("regression", {"method": "group_kfold", "n_splits": 3},
            training={"early_stopping": {"enabled": True, "inner_valid": {
                "method": "group_holdout", "ratio": 0.2, "random_state": 7}}}))
    run("inner time_holdout", {"TimeHoldoutInnerValid.ratio": 0.2},
        fit("regression", {"method": "time_series", "n_splits": 3},
            training={"early_stopping": {"enabled": True, "inner_valid": {
                "method": "time_holdout", "ratio": 0.2}}}))
    run("metrics (binary)", {"ECE.n_bins": 5, "PrecisionAtK.k": 20},
        fit("binary", {"method": "kfold", "n_splits": 3},
            evaluation={"metrics": [{"ece": {"n_bins": 5}}, {"precision_at_k": {"k": 20}}]}))
    run("metrics (regression)", {"HuberLoss.delta": 2.0},
        fit("regression", {"method": "kfold", "n_splits": 3},
            evaluation={"metrics": [{"huber": {"delta": 2.0}}]}))

    def feval_fit() -> object:
        df = frame("binary")
        cfg = config("binary", {"method": "kfold", "n_splits": 3})
        cfg["model"]["params"]["metric"] = [{"precision_at_k": {"k": 15}}]
        return Model(cfg).fit(data=df)

    run("metric via feval", {"PrecisionAtK.k": 15}, feval_fit)
    for name, cls_name in (("platt", "PlattCalibrator"), ("isotonic", "IsotonicCalibrator"),
                           ("beta", "BetaCalibrator")):
        params = {"tol": 1e-7} if name != "isotonic" else {"num_leaves": 7}
        run(f"calibration {name}", {f"{cls_name}.params": params},
            fit("binary", {"method": "kfold", "n_splits": 3},
                calibration={"method": name, "params": params}))
    run("unseen_policy", {"CategoricalEncoder.unseen_policy": "nan",
                          "NativeFeaturePipeline.unseen_policy": "nan"},
        fit("regression", {"method": "kfold", "n_splits": 3}, features={
            "exclude": ["g", "t", "blk"], "unseen_policy": "nan"}))

    def tune() -> object:
        df = frame("regression")
        cfg = config("regression", {"method": "kfold", "n_splits": 3},
                     training={"seed": 7},
                     tuning={"optuna": {"params": {"n_trials": 2, "direction": "maximize",
                                                   "timeout": 600.0}}})
        cfg["evaluation"] = {"metrics": ["r2"]}
        return Model(cfg).tune(data=df, progress_callback=_cb, study_name="probe")

    run("tune", {"Tuner.n_trials": 2, "Tuner.direction": "maximize", "Tuner.timeout": 600.0,
                 "Tuner.seed": 7, "Tuner.progress_callback": _cb, "Tuner.study_name": "probe"},
        tune)


def _cb(info: Any) -> None:  # noqa: ARG001
    return None


if __name__ == "__main__":
    main()
