"""PR 8b: what the three tuning surfaces return on a tuned model after ``load()``.

H-0086 persists the tuning result's ``best_*`` overlay and leaves out the trial
list, the rounds and the boundary report. This records what each surface then
presents, for the follow-up issue the census (``pr8b_load_census.py``) points to.

Run:

    uv run python docs/audits/2026-09-defect-discovery/instruments/pr8b_tuning_surfaces_after_load.py
"""

from __future__ import annotations

import sys
import tempfile
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from lizyml import Model  # noqa: E402
from lizyml.core.exceptions import LizyMLError  # noqa: E402
from tests._helpers import make_binary_df, make_config  # noqa: E402

warnings.simplefilter("ignore")


def describe(model: Model) -> dict[str, str]:
    out: dict[str, str] = {}
    table = model.tuning_table()
    out["tuning_table"] = f"DataFrame shape {table.shape}, columns {list(table.columns)[:4]}"
    fig = model.tuning_plot()
    out["tuning_plot"] = (
        f"Figure with {len(fig.data)} trace(s), "
        f"points per trace {[len(t.x) if t.x is not None else 0 for t in fig.data]}"
    )
    try:
        model.boundary_table()
        out["boundary_table"] = "returns a table"
    except LizyMLError as exc:
        out["boundary_table"] = f"raises {exc.code.name}: {exc.user_message}"
    return out


def main() -> int:
    cfg = make_config("binary", n_estimators=10, n_splits=2, num_threads=1, tuning_n_trials=2)
    model = Model(cfg, data=make_binary_df(n=200))
    model.tune()
    model.tune(resume=True, expand_boundary=True)
    model.fit()
    with tempfile.TemporaryDirectory() as tmp:
        model.export(Path(tmp) / "a")
        loaded = Model.load(Path(tmp) / "a")
        for label, m in (("fitted", model), ("loaded", loaded)):
            print(f"[{label}]")
            for surface, text in describe(m).items():
                print(f"  {surface}: {text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
