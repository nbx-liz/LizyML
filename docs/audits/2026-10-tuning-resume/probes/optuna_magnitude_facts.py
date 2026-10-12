"""Optuna facts behind the H-0124 amendment 1 magnitude rule (#350), executed.

Run: .venv/bin/python docs/audits/2026-10-tuning-resume/probes/optuna_magnitude_facts.py
Output is saved next to this file as optuna_magnitude_facts.txt.

Optuna samples an ``IntDistribution`` through float, widening its bounds by
half a step (``optuna/_transform.py``: ``low - half_step``, ``high +
half_step``). A half-integer is exact as a float only below ``2**52`` (the
spacing of floats is 1 in ``[2**52, 2**53)`` and 2 from ``2**53``), so
``2**52`` is the largest magnitude at which Optuna's int transform is exact;
from ``2**53`` adjacent integers merge.
"""

from __future__ import annotations

import warnings

import optuna

optuna.logging.set_verbosity(optuna.logging.ERROR)
warnings.simplefilter("ignore")
print("optuna", optuna.__version__)


def ints(low: int, high: int, sampler: optuna.samplers.BaseSampler, n: int = 400) -> str:
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


print("== half-integers as floats (Optuna widens int bounds by step/2)")
for exp in (51, 52, 53):
    x = 2**exp
    print(f"- 2**{exp}: float(x + 0.5) == x + 0.5 exactly: {float(x) + 0.5 == x + 0.5 and (float(x) + 0.5) - float(x) == 0.5}")

print("== suggest_int over width-10 ranges (RandomSampler / TPESampler)")
ranges = {
    "[2**31, 2**31+10]": (2**31, 2**31 + 10),
    "[2**52-10, 2**52] (the limit, inside)": (2**52 - 10, 2**52),
    "[-2**52, -2**52+10] (the limit, negative)": (-(2**52), -(2**52) + 10),
    "[2**52, 2**52+10]": (2**52, 2**52 + 10),
    "[2**52+1000, 2**52+1010]": (2**52 + 1000, 2**52 + 1010),
    "[2**53, 2**53+10]": (2**53, 2**53 + 10),
    "[-2**53-10, -2**53]": (-(2**53) - 10, -(2**53)),
    "[2**54, 2**54+10]": (2**54, 2**54 + 10),
    "[2**60, 2**60+10]": (2**60, 2**60 + 10),
    "[2**100, 2**100+10]": (2**100, 2**100 + 10),
}
for label, (low, high) in ranges.items():
    for name, sampler in (
        ("Random", optuna.samplers.RandomSampler(seed=0)),
        ("TPE", optuna.samplers.TPESampler(seed=0)),
    ):
        print(f"- {label}, {name}: {ints(low, high, sampler)}")
for name, sampler in (
    ("Random", optuna.samplers.RandomSampler(seed=0)),
    ("TPE", optuna.samplers.TPESampler(seed=0)),
):
    print(f"- [10**309, 2*10**309], {name}: {ints(10**309, 2 * 10**309, sampler, n=3)}")

print("== suggest_float where each bound is finite (RandomSampler)")
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
