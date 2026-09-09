"""Measure whether a search-space value reaches training normalised.

Round 29 of the PR-2 review found the one seam where the three proposals of that
pull request do not compose: H-0095 normalises a value written at four entrances,
and H-0096 refuses a duplicate spelling at five positions including
``tuning.optuna.space`` -- but a value *sampled* from a search dimension is
overlaid onto the trial parameters after the entrance, so nothing normalises it.
It reaches the exit assertion in the adapter as the type it was written as.

The failure is closed: the assertion refuses, and no Booster trains. That is why
this is filed rather than fixed inside PR 2 -- the direction is a valid input
being refused, not an invalid one admitted.

Run it from the repository root::

    uv run python docs/audits/2026-09-defect-discovery/instruments/space_choice_normalisation.py

Both cases are printed, because the numpy case only means something beside the
plain-float control: without the control, a broken tuning setup would look the
same.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "tests"))

from _helpers import make_binary_df, make_config  # noqa: E402

from lizyml.core.exceptions import LizyMLError  # noqa: E402
from lizyml.core.model import Model  # noqa: E402


def run(choice: object, label: str) -> None:
    """Tune over a single categorical dimension holding exactly ``choice``."""
    config = make_config("binary", n_estimators=3, n_splits=2, tuning_n_trials=2)
    config["tuning"]["optuna"]["space"] = {
        "eta": {"type": "categorical", "choices": [choice], "category": "model"},
    }

    try:
        model = Model(config, data=make_binary_df(n=160))
        model.tune()
    except LizyMLError as error:
        print(f"{label}: refused, code={error.code.name}")
        cause = error.__cause__
        while cause is not None:
            code = getattr(cause, "code", None)
            name = getattr(code, "name", code)
            print(f"{label}:   caused by {type(cause).__name__} code={name}")
            cause = cause.__cause__
    else:
        best = model._tuning_result  # noqa: SLF001 -- no public reader for this
        print(f"{label}: tuned, best_model_params={best.best_model_params}")


def main() -> int:
    print(f"numpy {np.__version__}")
    run(np.float64(0.5), "numpy scalar   ")
    run(0.5, "plain float    ")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
