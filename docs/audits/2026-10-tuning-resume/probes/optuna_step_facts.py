"""Optuna facts H-0124 (#350, search-dimension `step`) relies on, executed.

Run: .venv/bin/python docs/audits/2026-10-tuning-resume/probes/optuna_step_facts.py
Output is saved next to this file as optuna_step_facts.txt.
"""

from __future__ import annotations

import warnings
from decimal import Decimal

import optuna
from optuna.distributions import (
    FloatDistribution,
    IntDistribution,
    check_distribution_compatibility,
)

optuna.logging.set_verbosity(optuna.logging.ERROR)
print("optuna", optuna.__version__)


def fact(label: str, fn) -> None:  # type: ignore[no-untyped-def]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            result = fn()
            outcome = f"ok: {result!r}"
        except Exception as exc:  # noqa: BLE001 - the probe reports any outcome
            outcome = f"{type(exc).__name__}: {exc}"
    names = [f"{w.category.__name__}: {str(w.message)[:90]}" for w in caught]
    print(f"- {label}: {outcome}" + (f" | warnings: {names}" if names else ""))


# 1. A width that is not a multiple of step is truncated, with a warning.
fact("FloatDistribution(0.1, 1.0, step=0.4).high", lambda: FloatDistribution(0.1, 1.0, step=0.4).high)
fact("IntDistribution(0, 10, step=3).high", lambda: IntDistribution(0, 10, step=3).high)

# 2. log with step.
fact("FloatDistribution(0.001, 0.1, log=True, step=0.001)", lambda: FloatDistribution(0.001, 0.1, log=True, step=0.001))
fact("IntDistribution(1, 64, log=True, step=2)", lambda: IntDistribution(1, 64, log=True, step=2))
fact("IntDistribution(1, 64, log=True, step=1)", lambda: IntDistribution(1, 64, log=True, step=1))

# 3. Compatibility checks ignore step; the distribution class and log matter.
fact(
    "compat Float step 0.1 vs step 0.2",
    lambda: check_distribution_compatibility(
        FloatDistribution(0.0, 1.0, step=0.1), FloatDistribution(0.0, 1.0, step=0.2)
    ),
)
fact(
    "compat Int(log) vs Float(log)",
    lambda: check_distribution_compatibility(
        IntDistribution(1, 64, log=True), FloatDistribution(1.0, 64.0, log=True)
    ),
)
fact(
    "compat Int step 1 vs Int step 2",
    lambda: check_distribution_compatibility(IntDistribution(0, 10, step=1), IntDistribution(0, 10, step=2)),
)


# 4. Native float step values carry binary-float noise; Decimal(str) is exact.
def native_noise() -> str:
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    off = set()
    for _ in range(400):
        trial = study.ask()
        v = trial.suggest_float("x", 0.1, 1.0, step=0.1)
        k = (Decimal(str(v)) / Decimal("0.1")).to_integral_value()
        canonical = float(k * Decimal("0.1"))
        if v != canonical:
            off.add((v, canonical))
        study.tell(trial, 0.0)
    return f"{len(off)} distinct sampled values differ from float(k*Decimal('0.1')): {sorted(off)[:4]}"


fact("native suggest_float(0.1, 1.0, step=0.1) noise", native_noise)
fact("float remainder 0.3 % 0.1", lambda: 0.3 % 0.1)
fact("Decimal('0.3') % Decimal('0.1')", lambda: Decimal("0.3") % Decimal("0.1"))
