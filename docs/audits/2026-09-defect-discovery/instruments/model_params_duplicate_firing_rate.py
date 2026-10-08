"""Measure how often the `model.params` same-layer refusal would fire.

ARCHIVED EVIDENCE -- does not run after H-0096. It imports
`lizyml.core.value_equality.values_differ`, which H-0096 deleted; the refusal it
measured now fires on any duplicate spelling rather than only on differing
values. Reproducing its number needs a checkout at 251353d or earlier.


The refusal added for review round 11 changes what an existing config does, so
the Change Gate asks for a measured firing rate before it ships rather than an
estimate from reading the condition.

Installed as a pytest plugin, it records every `model.params` dict this
repository's own suite constructs and reports how many carry one parameter under
two spellings -- and, of those, how many carry two *different* values, which is
the shape that is refused.

    uv run pytest -p docs.audits...instruments.model_params_duplicate_firing_rate

or, from the repository root:

    .venv/bin/python -m pytest tests -q --no-cov \\
        -p docs.audits.2026-09-defect-discovery.instruments.<this module>

The path is not importable as a package, so it is loaded by `-p` with a
`conftest`-style path instead in practice; the counters below are what matters.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from lizyml.core.value_equality import values_differ
from lizyml.estimators.lgbm.param_names import LGBM_CANONICAL_NAME

COUNTS: Counter[str] = Counter()
CONFLICTS: list[dict[str, Any]] = []


def observe(params: dict[str, Any]) -> None:
    """Classify one `model.params` dict."""
    COUNTS["configs with model.params"] += 1
    grouped: dict[str, dict[str, Any]] = {}
    for name, value in params.items():
        canonical = LGBM_CANONICAL_NAME.get(name, name)
        grouped.setdefault(canonical, {})[name] = value

    duplicated = {c: w for c, w in grouped.items() if len(w) > 1}
    if not duplicated:
        return
    COUNTS["carrying one parameter under two spellings"] += 1

    conflicting = {
        canonical: written
        for canonical, written in duplicated.items()
        if any(
            values_differ(value, next(iter(written.values())))
            for value in written.values()
        )
    }
    if conflicting:
        COUNTS["refused: two spellings, different values"] += 1
        CONFLICTS.append(conflicting)


def report() -> str:
    total = COUNTS["configs with model.params"]
    refused = COUNTS["refused: two spellings, different values"]
    lines = [f"{name}: {count}" for name, count in sorted(COUNTS.items())]
    lines.append(f"Firing rate: {refused}/{total} of configs carrying model.params")
    if CONFLICTS:
        lines.append(f"conflicts: {CONFLICTS}")
    return "\n".join(lines)
