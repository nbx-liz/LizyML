"""PR 8b (#281): which reporting-surface readings do not survive export -> load?

Prints the census ``tests/test_persistence/_load_census.py`` defines -- the same
definition ``test_reporting_surfaces_survive_load.py`` enforces, so the printed
numbers and the asserted ones cannot drift apart. See that module for the
population (``INVENTORY``, every public name on ``Model``), the configurations,
the lifecycles, and the bound (surfaces are called with their defaults).

Run:

    uv run python docs/audits/2026-09-defect-discovery/instruments/pr8b_load_census.py

Prints one line per (configuration, lifecycle, surface) whose readings differ
(``importance_gain`` counts as differing here when not bit-identical, with its
largest relative error), then the counts. Exits 0 always; it measures.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from tests.test_persistence import _load_census as census  # noqa: E402


def describe(before: tuple[str, Any], after: tuple[str, Any]) -> str:
    if before[0] != after[0] or before[0] == "raises":
        return f"{before[0]}:{before[1] if before[0] == 'raises' else ''} -> " + (
            f"{after[0]}:{after[1] if after[0] == 'raises' else ''}"
        )
    x, y = before[1], after[1]
    if isinstance(x, dict):
        keys = sorted(k for k in x.keys() | y.keys() if not census.same(x.get(k), y.get(k)))
        return "keys " + ", ".join(f"{k}: {x.get(k)!r} -> {y.get(k)!r}" for k in keys)
    if isinstance(x, pd.DataFrame) and x.index.equals(y.index) and x.shape == y.shape:
        rows = [i for i in x.index if not census.same(x.loc[[i]], y.loc[[i]])]
        return "rows " + ", ".join(
            f"{i}: {x.loc[i].tolist()} -> {y.loc[i].tolist()}" for i in rows
        )
    if isinstance(x, pd.DataFrame):
        return f"frame shape {x.shape} -> {y.shape}"
    return "value differs"


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        cells = census.collect(Path(tmp))

    by_surface: dict[str, int] = {}
    worst_gain = 0.0
    for entry in cells:
        config, lifecycle, surface = entry["cell"]
        before, after = entry["before"], entry["after"]
        if census.same(before, after):
            continue
        by_surface[surface] = by_surface.get(surface, 0) + 1
        if surface == "importance_gain" and before[0] == after[0] == "ok":
            for key, value in before[1].items():
                if value:
                    worst_gain = max(worst_gain, abs(after[1][key] - value) / abs(value))
        print(f"{config:<17} {lifecycle:<18} {surface:<27} {describe(before, after)}")

    print()
    print(
        f"cells: {len(cells)} ({len(census.CONFIGURATIONS)} configurations x "
        f"{len(census.LIFECYCLES)} lifecycles x {len(census.SURFACES)} surfaces); "
        f"differing: {sum(by_surface.values())}"
    )
    for surface, n in sorted(by_surface.items()):
        print(f"  {surface}: {n}")
    print(f"largest relative importance_gain error: {worst_gain:.3e} "
          f"(declared bound {census.GAIN_RTOL})")
    print("cells where the fitted model returned a reading (exercised):")
    for surface in census.SURFACES:
        n = sum(1 for e in cells if e["cell"][2] == surface and e["before"][0] == "ok")
        print(f"  {surface}: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
