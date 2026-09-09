"""Measure the *tolerance* branch: same-layer duplicate spellings with EQUAL values.

The round-11 instrument reported only the refused (different-value) count.
The question D13's analysis needs is the other branch: over the shipped suite,
how often does one layer name one parameter under two accepted spellings with
values the gate calls equal?  That is the population a
"refuse any duplicate spelling" rule would newly break.

Both call sites of `values_differ` are wrapped:
  * `_model_factories.check_duplicate_identities`  (model.params / calibration.params)
  * `lgbm.adapter._pop_by_identity`                (adapter dedupe)

Run from the repository root, with this directory on PYTHONPATH:

    PYTHONPATH=docs/audits/2026-09-defect-discovery/instruments \\
    DUP_TOLERANCE_OUT=/path/to/report.txt \\
    .venv/bin/python -m pytest tests -q --no-cov \\
        -p duplicate_tolerance_firing_rate

Measured at 251353d on 2026-09-09: 51 same-layer duplicate spellings, 14
refused (different values), 37 tolerated (equal values) -- and all 37 come
from tests/test_core/test_fit_params_override.py, a file this PR adds. The
tolerance branch has a firing rate of 0 over the pre-existing population.
"""

from __future__ import annotations

import os
from collections import Counter
from typing import Any

COUNTS: Counter[str] = Counter()
EQUAL_DUPES: list[Any] = []


def _classify(params: dict[str, Any], canonical_of, where: str) -> None:
    from lizyml.core.value_equality import values_differ

    COUNTS[f"{where}: dicts observed"] += 1
    grouped: dict[str, dict[str, Any]] = {}
    for name, value in params.items():
        grouped.setdefault(canonical_of(name), {})[name] = value
    dupes = {c: w for c, w in grouped.items() if len(w) > 1}
    if not dupes:
        return
    COUNTS[f"{where}: one parameter under two spellings"] += 1
    for canonical, written in dupes.items():
        first = next(iter(written.values()))
        try:
            differs = any(values_differ(v, first) for v in written.values())
        except Exception:
            COUNTS[f"{where}: comparison raised"] += 1
            continue
        if differs:
            COUNTS[f"{where}: REFUSED (different values)"] += 1
        else:
            COUNTS[f"{where}: TOLERATED (equal values)"] += 1
            EQUAL_DUPES.append((os.environ.get("PYTEST_CURRENT_TEST", "?").split(" ")[0], where, canonical, dict(written)))


def pytest_configure(config):  # noqa: ARG001
    from lizyml.core import _model_factories
    from lizyml.estimators.lgbm import adapter

    original_check = _model_factories.check_duplicate_identities

    def wrapped_check(provider, params, *, surface):
        if params:
            canonical = provider.canonical_param_names(params)
            _classify(params, lambda n: canonical[n], f"factories[{surface}]")
        return original_check(provider, params, surface=surface)

    _model_factories.check_duplicate_identities = wrapped_check

    original_pop = adapter._pop_by_identity

    def wrapped_pop(user_params, canonical, *args, **kwargs):
        accepted = set(adapter.accepted_spellings(canonical))
        supplied = {n: v for n, v in user_params.items() if n in accepted}
        if supplied:
            _classify(supplied, lambda _n: canonical, "adapter")
        return original_pop(user_params, canonical, *args, **kwargs)

    adapter._pop_by_identity = wrapped_pop


def pytest_terminal_summary(terminalreporter, *args, **kwargs):  # noqa: ARG001
    lines = [f"{k}: {v}" for k, v in sorted(COUNTS.items())]
    lines.append(f"EQUAL-VALUE DUPLICATES: {len(EQUAL_DUPES)}")
    for item in EQUAL_DUPES[:60]:
        lines.append(f"  {item}")
    text = "\n".join(lines)
    terminalreporter.write_line("=== DUP TOLERANCE MEASUREMENT ===")
    terminalreporter.write_line(text)
    out = os.environ.get("DUP_TOLERANCE_OUT")
    if out:
        with open(out, "w") as fh:
            fh.write(text + "\n")
