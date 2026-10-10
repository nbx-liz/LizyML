"""Probe resume / storage behaviour against objective changes (read-only)."""

from __future__ import annotations

import copy
import logging
import sys
import tempfile
import warnings

sys.path.insert(0, "/home/rem/repos/LizyML")
warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

import optuna  # noqa: E402

optuna.logging.set_verbosity(optuna.logging.ERROR)

from lizyml import Model  # noqa: E402
from tests._helpers import make_regression_df  # noqa: E402

SPACE = {
    "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
    "n_estimators": {"type": "int", "low": 5, "high": 20},
}


def cfg(metric: str = "rmse", space: dict | None = None, auto_nl: bool = True) -> dict:
    return {
        "config_version": 1,
        "task": "regression",
        "data": {"target": "target"},
        "model": {"name": "lgbm", "auto_num_leaves": auto_nl},
        "split": {"method": "kfold", "n_splits": 3},
        "evaluation": {"metrics": [metric]},
        "training": {"early_stopping": {"enabled": False}},
        "tuning": {
            "optuna": {
                "params": {"n_trials": 4},
                "space": space if space is not None else SPACE,
                "space_mode": "replace",
            }
        },
    }


d1 = make_regression_df(n=150, seed=0)
d2 = make_regression_df(n=150, seed=1)
d2["target"] = d2["target"] * 100.0  # other scale -> rmse scale differs

print("== P1: in-process resume after data change ==")
m = Model(cfg())
r1 = m.tune(data=d1)
print("round1 best_score", r1.best_score)
r2 = m.tune(data=d2, resume=True, expand_boundary=False)
scores = [(t.number, t.round, round(t.score, 4)) for t in r2.trials]
print("trials (number, round, score)", scores)
print("round2 best_score", r2.best_score, "best_params", r2.best_params)
print("best trial is from round", [t.round for t in r2.trials if t.score == r2.best_score])

print("== P2: in-process resume after metric change (rmse->mae, same direction) ==")
m = Model(cfg("rmse"))
r1 = m.tune(data=d1)
m._cfg.evaluation.metrics = ["mae"]
r2 = m.tune(data=d1, resume=True, expand_boundary=False)
print("metric_name", r2.metric_name, "best_score", r2.best_score)
print("rounds", [(t.round, round(t.score, 4)) for t in r2.trials])

print("== P3: resume after config space change (space ignored) ==")
m = Model(cfg())
m.tune(data=d1)
m._cfg.tuning.optuna.space = {"learning_rate": {"type": "float", "low": 0.2, "high": 0.25}}
r = m.tune(data=d1, resume=True, expand_boundary=False)
print("space names", [(dim.name, getattr(dim, "low", None), getattr(dim, "high", None)) for dim in m._space])

print("== P4: storage reattach, new Model, different metric+space, resume=False ==")
tmp = tempfile.mkdtemp()
url = f"sqlite:///{tmp}/s.db"
ma = Model(cfg("rmse"))
ra = ma.tune(data=d1, storage=url, study_name="job")
mb = Model(
    cfg(
        "mae",
        space={"num_leaves_ratio": {"type": "float", "low": 0.5, "high": 1.0, "category": "smart"}},
    )
)
try:
    rb = mb.tune(data=d1, storage=url, study_name="job")
    print("n trials", len(rb.trials), "rounds of trials", sorted({t.round for t in rb.trials}))
    print("best_model", rb.best_model_params, "best_smart", rb.best_smart_params)
    print("metric", rb.metric_name, "best_score", rb.best_score)
    try:
        mb.fit(data=d1)
        print("fit after reattach: OK")
    except Exception as exc:  # noqa: BLE001
        print("fit after reattach raised", type(exc).__name__, str(exc)[:200])
except Exception as exc:  # noqa: BLE001
    print("reattach raised", type(exc).__name__, str(exc)[:300])

print("== P5: resume=True passes storage? (in-process study reused) ==")
m = Model(cfg())
m.tune(data=d1)
url2 = f"sqlite:///{tmp}/s2.db"
m.tune(data=d1, resume=True, storage=url2, study_name="other", expand_boundary=False)
print("study storage type", type(m._study._storage).__name__)
print("persisted studies in url2", [s.study_name for s in optuna.get_all_study_summaries(url2)])

print("== P6: load -> tune(resume=True) ==")
m = Model(cfg())
m.tune(data=d1)
m.fit(data=d1)
out = tempfile.mkdtemp() + "/m"
m.export(out)
lm = Model.load(out)
try:
    lm.tune(data=d1, resume=True)
except Exception as exc:  # noqa: BLE001
    print("raised", type(exc).__name__, getattr(exc, "code", None))

print("== P7: auto_num_leaves true->false then resume; ratio dim keeps sampling ==")
sp = {
    "num_leaves_ratio": {"type": "float", "low": 0.5, "high": 1.0, "category": "smart"},
    "n_estimators": {"type": "int", "low": 5, "high": 20},
}
m = Model(cfg(space=sp, auto_nl=True))
m.tune(data=d1)
m._cfg.model.auto_num_leaves = False
try:
    r = m.tune(data=d1, resume=True, expand_boundary=False)
    print("resumed; best_smart", r.best_smart_params)
except Exception as exc:  # noqa: BLE001
    print("raised", type(exc).__name__, str(exc)[:200])
