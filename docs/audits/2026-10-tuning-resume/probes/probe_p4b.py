"""P4b: persisted study written under mae, reattached under rmse + other space."""

from __future__ import annotations

import logging
import sys
import tempfile
import warnings

sys.path.insert(0, "/tmp/claude-1000/-home-rem-repos-LizyML/d536b2c8-f474-4f26-86db-5077863e45d9/scratchpad")
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


def cfg(metric: str, space: dict) -> dict:
    return {
        "config_version": 1,
        "task": "regression",
        "data": {"target": "target"},
        "model": {"name": "lgbm"},
        "split": {"method": "kfold", "n_splits": 3},
        "evaluation": {"metrics": [metric]},
        "training": {"early_stopping": {"enabled": False}},
        "tuning": {
            "optuna": {
                "params": {"n_trials": 3},
                "space": space,
                "space_mode": "replace",
            }
        },
    }


d1 = make_regression_df(n=150, seed=0)
url = f"sqlite:///{tempfile.mkdtemp()}/s.db"
seen = []
Model(cfg("mae", SPACE)).tune(data=d1, storage=url, study_name="job")
mb = Model(
    cfg(
        "rmse",
        {"num_leaves_ratio": {"type": "float", "low": 0.5, "high": 1.0, "category": "smart"}},
    )
)
rb = mb.tune(
    data=d1, storage=url, study_name="job", progress_callback=lambda i: seen.append(i.current_trial)
)
print("trial scores", [(t.number, round(t.score, 4), sorted(t.params)) for t in rb.trials])
print("metric", rb.metric_name, "best_score", rb.best_score)
print("best_model", rb.best_model_params, "best_smart", rb.best_smart_params)
print("progress current_trial values", seen, "rounds", rb.rounds and [(r.round, r.n_trials) for r in rb.rounds])
mb.fit(data=d1)
print("fit OK; adapter lr/n:", mb.params_table().to_dict() if False else "")
params, _ = mb._merge_params(mb._provider)
print("fit merged params learning_rate", params.get("learning_rate"), "n_estimators", params.get("n_estimators"))
