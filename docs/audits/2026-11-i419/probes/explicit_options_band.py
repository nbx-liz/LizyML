"""Seeded TPE sequence with the H-0149 explicit options, plus the warnings construction emits."""

import hashlib
import json
import math
import warnings

import optuna

optuna.logging.set_verbosity(optuna.logging.ERROR)


def objective(trial: optuna.Trial) -> float:
    lr = trial.suggest_float("learning_rate", 1e-3, 0.3, log=True)
    ff = trial.suggest_float("feature_fraction", 0.5, 1.0)
    nl = trial.suggest_int("num_leaves", 8, 256, log=True)
    md = trial.suggest_int("min_data_in_leaf", 5, 100)
    bt = trial.suggest_categorical("boosting", ["gbdt", "dart"])
    pen = 0.1 if bt == "dart" else 0.0
    return (math.log(lr) + 3) ** 2 + (ff - 0.8) ** 2 + (math.log(nl) - 4) ** 2 / 10 + (md - 30) ** 2 / 1000 + pen


with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    sampler = optuna.samplers.TPESampler(seed=42, multivariate=True, constant_liar=True)
ctor_warnings = sorted({f"{w.category.__name__}: {str(w.message)[:80]}" for w in caught})

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=40)
run_warnings = sorted({f"{w.category.__name__}: {str(w.message)[:80]}" for w in caught})

seq = [{k: (round(v, 10) if isinstance(v, float) else v) for k, v in t.params.items()} for t in study.trials]
print(json.dumps({
    "optuna": optuna.__version__,
    "sha": hashlib.sha256(json.dumps(seq, sort_keys=True).encode()).hexdigest()[:16],
    "ctor_warnings": ctor_warnings,
    "run_warnings": run_warnings,
}))
