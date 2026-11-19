"""Seeded TPESampler trial sequences under the installed Optuna, for five sampler variants.

Run with an interpreter that has one Optuna version installed (see ../README.md). Prints one
JSON line: the Optuna and NumPy versions, and per variant the sha256 prefix of the 40-trial
parameter sequence plus the parameters of trial 9 (last startup trial) and trial 10 (first
TPE trial).
"""

import hashlib
import json
import math
import warnings

import numpy
import optuna

optuna.logging.set_verbosity(optuna.logging.ERROR)
warnings.filterwarnings("ignore")

N_TRIALS = 40
SEED = 42


def objective(trial: optuna.Trial) -> float:
    lr = trial.suggest_float("learning_rate", 1e-3, 0.3, log=True)
    ff = trial.suggest_float("feature_fraction", 0.5, 1.0)
    nl = trial.suggest_int("num_leaves", 8, 256, log=True)
    md = trial.suggest_int("min_data_in_leaf", 5, 100)
    bt = trial.suggest_categorical("boosting", ["gbdt", "dart"])
    pen = 0.1 if bt == "dart" else 0.0
    return (
        (math.log(lr) + 3) ** 2
        + (ff - 0.8) ** 2
        + (math.log(nl) - 4) ** 2 / 10
        + (md - 30) ** 2 / 1000
        + pen
    )


VARIANTS = {
    "default": {},
    "v4_options": {"multivariate": False, "constant_liar": False},
    "v5_options": {"multivariate": True, "constant_liar": True},
    "multivariate_only": {"multivariate": True, "constant_liar": False},
    "constant_liar_only": {"multivariate": False, "constant_liar": True},
}


def run(kwargs: dict) -> list[dict]:
    sampler = optuna.samplers.TPESampler(seed=SEED, **kwargs)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=N_TRIALS)
    return [
        {k: (round(v, 10) if isinstance(v, float) else v) for k, v in t.params.items()}
        for t in study.trials
    ]


out: dict = {"optuna": optuna.__version__, "numpy": numpy.__version__}
for name, kw in VARIANTS.items():
    seq = run(kw)
    digest = hashlib.sha256(json.dumps(seq, sort_keys=True).encode()).hexdigest()[:16]
    out[name] = {"sha": digest, "trial9": seq[9], "trial10": seq[10]}
print(json.dumps(out, sort_keys=True))
