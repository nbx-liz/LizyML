"""H-0143: the H-0136 grid population, verdict against the real serializers.

For every column of ``tests/test_persistence/test_h0136_grid.py`` (the H-0133
samples alone and in pairs, beside ``"x"``), compares the real joblib and
json outcomes with two verdicts: today's ``state_changes`` and the H-0143
prototype. Prints the mismatches of each, grouped by the column's value types.

    PYTHONPATH=. python -B docs/audits/2026-10-h0143/probes/grid_census.py
"""

from __future__ import annotations

import sys
import tempfile
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from prototype import prototype_state_changes  # noqa: E402

from lizyml.features.encoders.categorical_encoder import CategoricalEncoder  # noqa: E402
from lizyml.persistence._category_preflight import state_changes  # noqa: E402
from tests.test_persistence import test_h0136_grid as grid  # noqa: E402

warnings.simplefilter("ignore")


class _Factory:
    def mktemp(self, name: str) -> Path:
        return Path(tempfile.mkdtemp(prefix=name))


def _kind(column: tuple, verdict: set, real: set) -> str:
    types = " + ".join(sorted(type(v).__name__ for v in column))
    return f"{types}: verdict {sorted(verdict)} real {sorted(real)}"


def main() -> None:
    print(f"pandas {pd.__version__} numpy {np.__version__}")
    module, project = grid.generated.__wrapped__(_Factory())
    tmp = Path(tempfile.mkdtemp())
    today: Counter[str] = Counter()
    proposed: Counter[str] = Counter()
    changed = 0
    judged = 0
    for column in grid._COLUMNS:
        train = pd.DataFrame({"c": pd.Series([*column, *column, "x"], dtype=object)})
        try:
            encoder = CategoricalEncoder(unseen_policy="error").fit(train, ["c"])
            state = encoder.get_state()
            df = pd.DataFrame({"c": pd.Series(list(state["categories"]["c"]), dtype=object)})
            want = grid._codes(encoder.transform(df))
        except Exception:  # noqa: BLE001 -- the fit fails: not a case, as in the test
            continue
        judged += 1
        old = {c.serializer for c in state_changes({"encoder": state})}
        new = {c.serializer for c in prototype_state_changes({"encoder": state})}
        changed += old != new
        want_mode = grid._mode_code(encoder, df)
        real = set()
        if grid._joblib_breaks(encoder, df, want, want_mode, tmp):
            real.add("joblib")
        if grid._json_breaks(module, project, state, df, want, want_mode):
            real.add("json")
        if old != real:
            today[_kind(column, old, real)] += 1
        if new != real:
            proposed[_kind(column, new, real)] += 1
    print(f"judged {judged}; verdict changed by H-0143: {changed}")
    for label, counts in (("today", today), ("H-0143", proposed)):
        print(f"{label}: {sum(counts.values())} mismatches")
        for kind, n in counts.most_common():
            print(f"  {n:5d}  {kind}")


if __name__ == "__main__":
    main()
