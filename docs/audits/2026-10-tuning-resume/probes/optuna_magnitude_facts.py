"""Optuna facts behind the H-0124 amendment 1 magnitude rule (#350), executed.

Run: .venv/bin/python docs/audits/2026-10-tuning-resume/probes/optuna_magnitude_facts.py
Output is saved next to this file as optuna_magnitude_facts.txt.
"""

from __future__ import annotations

import warnings

import optuna

optuna.logging.set_verbosity(optuna.logging.ERROR)
warnings.simplefilter("ignore")
print("optuna", optuna.__version__)


def ints(low: int, high: int, sampler: optuna.samplers.BaseSampler, n: int = 300) -> str:
    study = optuna.create_study(sampler=sampler)
    out: list[int] = []
    for i in range(n):
        trial = study.ask()
        try:
            value = trial.suggest_int("n", low, high)
        except Exception as exc:  # noqa: BLE001 - the probe reports any outcome
            return f"{type(exc).__name__}: {str(exc)[:70]}"
        study.tell(trial, float(i % 7))
        out.append(value)
    inside = sum(low <= v <= high for v in out)
    return f"{inside}/{n} inside the range, {len(set(out))} distinct values of {high - low + 1}"


print("== suggest_int over a width-10 range at growing magnitude")
for exp in (31, 52, 53, 54, 60, 100):
    low = 2**exp
    for name, sampler in (
        ("Random", optuna.samplers.RandomSampler(seed=0)),
        ("TPE", optuna.samplers.TPESampler(seed=0)),
    ):
        print(f"- low 2**{exp}, {name}: {ints(low, low + 10, sampler)}")
for name, sampler in (
    ("Random", optuna.samplers.RandomSampler(seed=0)),
    ("TPE", optuna.samplers.TPESampler(seed=0)),
):
    print(f"- low 10**309, {name}: {ints(10**309, 2 * 10**309, sampler, n=3)}")

print("== suggest_float where each bound is finite")
for low, high, step in (
    (-1e308, 1e308, None),
    (-1.5e308, 1e308, 5e307),
    (-9e307, 9e307, None),
    (-8e307, 8e307, None),
):
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    trial = study.ask()
    try:
        value = trial.suggest_float("x", low, high, step=step)
        outcome = f"sampled {value!r}"
    except Exception as exc:  # noqa: BLE001
        outcome = f"{type(exc).__name__}: {exc}"
    print(f"- [{low!r}, {high!r}] step {step!r} (width {high - low!r}): {outcome}")
