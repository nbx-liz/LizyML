"""H-0128 policy 6 facts about Optuna: (1) a finished trial's user attributes
cannot be written; (2) an objective returning NaN, inf or a non-number ends as
FAIL without raising out of ``optimize``; (3) no ``fail_reason`` system attr."""

from __future__ import annotations

import math

import optuna

optuna.logging.set_verbosity(optuna.logging.ERROR)
print("optuna", optuna.__version__)

study = optuna.create_study()
study.optimize(lambda t: t.suggest_float("x", 0, 1), n_trials=1)
trial = study.trials[0]
try:
    study._storage.set_trial_user_attr(trial._trial_id, "k", "v")
    print("(1) writing a COMPLETE trial's user attr: allowed")
except Exception as exc:  # noqa: BLE001 - the probe reports what escapes
    print(f"(1) writing a COMPLETE trial's user attr: {type(exc).__name__}: {exc}")

for label, value in (("nan", math.nan), ("inf", math.inf), ("str", "abc"), ("None", None)):
    s = optuna.create_study()
    try:
        s.optimize(lambda t, v=value: (t.suggest_float("x", 0, 1), v)[1], n_trials=1)
        t = s.trials[0]
        print(f"(2) objective returns {label}: state={t.state.name} system_attrs={t.system_attrs}")
    except Exception as exc:  # noqa: BLE001
        print(f"(2) objective returns {label}: raised {type(exc).__name__}: {exc}")


def boom(t: optuna.Trial) -> float:
    t.suggest_float("x", 0, 1)
    raise ValueError("bad value")


s = optuna.create_study()
s.optimize(boom, n_trials=1, catch=(ValueError,))
print(f"(3) caught ValueError: state={s.trials[0].state.name} system_attrs={s.trials[0].system_attrs}")
