"""Multivariate TPE when a later round widens one dimension (LizyML re-tune boundary expansion).

Round 1: 15 trials with x in [0, 1], y in [0, 1]. Round 2 (same study, reloaded): 15 trials with
x in [0, 2] (widened), y unchanged. Reports warnings per round and whether x in round 2 is sampled
jointly with y (relative) or independently.
"""

import json
import warnings

import optuna
from optuna.samplers import TPESampler

optuna.logging.set_verbosity(optuna.logging.ERROR)


def make(xhigh: float):
    def objective(trial: optuna.Trial) -> float:
        x = trial.suggest_float("x", 0.0, xhigh)
        y = trial.suggest_float("y", 0.0, 1.0)
        return (x - 0.9) ** 2 + (y - 0.3) ** 2

    return objective


def sampler() -> TPESampler:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return TPESampler(seed=7, multivariate=True, constant_liar=True)


out = {"optuna": optuna.__version__}
storage = optuna.storages.InMemoryStorage()
study = optuna.create_study(storage=storage, study_name="s", sampler=sampler())
for label, xhigh in (("round1", 1.0), ("round2", 2.0)):
    study = optuna.load_study(storage=storage, study_name="s", sampler=sampler())
    s = study.sampler
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        study.optimize(make(xhigh), n_trials=15)
    last = study.trials[-1]
    relative = s.infer_relative_search_space(study, last)
    out[label] = {
        "warnings": sorted({f"{w.category.__name__}: {str(w.message)[:90]}" for w in caught}),
        "relative_space_at_end": sorted(relative),
    }
print(json.dumps(out, indent=1))
