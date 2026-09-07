"""Measure what round 12's two changes do to configs that already exist.

Both changes alter what an existing config does, so the Change Gate asks for a
measured rate rather than an estimate read off the condition.

  * ``calibration.params`` is now canonicalised before it reaches the
    calibrator, which merges it over its own defaults by spelling. A config
    carrying an **alias** of one of those defaults trained at the default and
    now trains at the value that was written.
  * ``values_differ`` normalises a non-text sequence to a ``list``, so one
    parameter written twice in two containers is no longer refused. A config
    carrying that shape was refused before this and trains now.

The population is every ``LizyMLConfig`` this repository's suite constructs,
recorded after a successful build so a config the schema already rejects never
enters it. Installed as a pytest plugin:

    .venv/bin/python -m pytest tests -q --no-cov -p <this module as a path>

Both counters are reported against the population each one is about, because a
rate over "every config" would understate both by the configs that carry no
parameters at all.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from lizyml.core.value_equality import values_differ
from lizyml.estimators.lgbm.param_names import LGBM_CANONICAL_NAME

COUNTS: Counter[str] = Counter()
#: What actually differed, so a non-zero rate can be read rather than trusted.
HITS: list[dict[str, Any]] = []

#: The calibrator pops these before LightGBM sees them, and ``num_boost_round``
#: is a LightGBM alias of ``num_iterations``, so canonicalising it would take
#: the key away from the code that pops it. Duplicated from
#: ``lizyml.calibration.isotonic`` on purpose: an instrument that imports the
#: subject cannot show the subject wrong about its own exclusions.
CALIBRATOR_OWN = frozenset(
    {"num_boost_round", "validation_ratio", "min_data_in_leaf_ratio"}
)


def observe_calibration(params: dict[str, Any], origin: str = "") -> None:
    """Classify one ``calibration.params`` dict.

    ``origin`` names the test that built it, so a hit added *by this change's
    own regression tests* can be told from one that was already in the
    population. A rate that counts the tests written to demonstrate the defect
    is not a rate over pre-existing configs.
    """
    COUNTS["configs with calibration.params"] += 1
    renamed = {
        name: LGBM_CANONICAL_NAME[name]
        for name in params
        if name not in CALIBRATOR_OWN
        and name in LGBM_CANONICAL_NAME
        and LGBM_CANONICAL_NAME[name] != name
    }
    if renamed:
        COUNTS["calibration.params carrying a non-canonical spelling"] += 1
        HITS.append(
            {"surface": "calibration.params", "renamed": renamed, "origin": origin}
        )


def observe_params(surface: str, params: dict[str, Any], origin: str = "") -> None:
    """Classify one parameter dict for the refusal this change lifts."""
    COUNTS[f"configs with {surface}"] += 1
    grouped: dict[str, dict[str, Any]] = {}
    for name, value in params.items():
        grouped.setdefault(LGBM_CANONICAL_NAME.get(name, name), {})[name] = value

    for canonical, written in grouped.items():
        if len(written) < 2:
            continue
        values = list(written.values())
        same_now = not any(values_differ(v, values[0]) for v in values)
        containers = {type(v).__name__ for v in values}
        if same_now and len(containers) > 1:
            COUNTS[f"{surface}: one value, two containers (was refused)"] += 1
            HITS.append(
                {
                    "surface": surface,
                    "parameter": canonical,
                    "written": written,
                    "origin": origin,
                }
            )


def report() -> str:
    lines = [f"{name}: {count}" for name, count in sorted(COUNTS.items())]
    lines.append(
        "Firing rate: "
        f"{COUNTS['calibration.params carrying a non-canonical spelling']}"
        f"/{COUNTS['configs with calibration.params']} of configs carrying "
        "calibration.params (canonicalisation changes the trained value)"
    )
    lines.append(
        "Firing rate: "
        f"{COUNTS['model.params: one value, two containers (was refused)']}"
        f"/{COUNTS['configs with model.params']} of configs carrying "
        "model.params (the lifted refusal)"
    )
    if HITS:
        lines.append(f"hits: {HITS}")
    return "\n".join(lines)
