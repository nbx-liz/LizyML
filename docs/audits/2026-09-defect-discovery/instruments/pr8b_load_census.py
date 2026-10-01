"""PR 8b (#281): which reporting-surface readings do not survive export -> load?

Every public reporting surface of ``Model`` is read twice: on the model that was
fitted, and on ``Model.load()`` of its export. A reading that differs is a value
the surface answers for that the artifact does not carry. Run over three tasks
and four lifecycles, so the census is not one configuration's answer.

The lifecycles are the orders in which ``fit`` and ``tune`` can precede an
export, plus one round trip through a second export, because a loaded model that
is exported again must carry what the first artifact carried:

* ``fit``                 -- no tuning at all
* ``tune_fit``            -- the fit consumed the tuning result (#281's row)
* ``fit_tune``            -- the artifact carries a tuning result no fit consumed
* ``tune_fit_reexport``   -- ``tune_fit``, then load -> export -> load
* ``tune_resume_fit``     -- ``tune``, ``tune(resume=True)``, ``fit``: the only
  order that produces a boundary report and a second round, so the only one in
  which ``boundary_table`` has a reading to lose

Run:

    uv run python docs/audits/2026-09-defect-discovery/instruments/pr8b_load_census.py

Prints one line per (task, lifecycle, surface) whose readings differ, then the
counts. Exits 0 always; it measures, it does not gate.
"""

from __future__ import annotations

import sys
import tempfile
import warnings
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from lizyml import Model  # noqa: E402
from lizyml.core.exceptions import LizyMLError  # noqa: E402
from tests._helpers import (  # noqa: E402
    make_binary_df,
    make_config,
    make_multiclass_df,
    make_regression_df,
)

warnings.simplefilter("ignore")

SPACE: dict[str, Any] = {
    "early_stopping_rounds": {
        "type": "categorical",
        "choices": [2],
        "category": "training",
    },
    "validation_ratio": {
        "type": "categorical",
        "choices": [0.45],
        "category": "training",
    },
}

#: Configurations, not only tasks: ``binary_calibrated`` is the one in which the
#: calibration surfaces have a reading at all.
DATA = {
    "regression": make_regression_df,
    "binary": make_binary_df,
    "binary_calibrated": make_binary_df,
    "multiclass": make_multiclass_df,
}
LIFECYCLES = ("fit", "tune_fit", "fit_tune", "tune_fit_reexport", "tune_resume_fit")

#: generate_code keyword arguments that are live objects, compared by type only.
OBJECT_KWARGS = {"model_adapter", "pipeline_state", "calibrator", "output_dir"}


def build(variant: str) -> Model:
    task = "binary" if variant == "binary_calibrated" else variant
    cfg = make_config(
        task, n_estimators=10, n_splits=2, num_threads=1, tuning_n_trials=1
    )
    cfg["training"]["early_stopping"] = {
        "enabled": True,
        "rounds": 7,
        "validation_ratio": 0.2,
    }
    cfg["tuning"]["optuna"]["space"] = dict(SPACE)
    if variant == "binary_calibrated":
        cfg["calibration"] = {"method": "platt"}
    return Model(cfg, data=DATA[variant](n=200))


def _export_code_kwargs(model: Model) -> dict[str, Any]:
    with mock.patch("lizyml.codegen.generator.generate_code") as generate:
        model.export_code("not-written")
    kwargs = dict(generate.call_args.kwargs)
    return {
        key: (type(value).__name__ if key in OBJECT_KWARGS else value)
        for key, value in kwargs.items()
    }


SURFACES: dict[str, Any] = {
    "evaluate_table": lambda m: m.evaluate_table(),
    "residuals": lambda m: m.residuals(),
    "confusion_matrix": lambda m: m.confusion_matrix(),
    "importance_split": lambda m: m.importance("split"),
    "importance_gain": lambda m: m.importance("gain"),
    "importance_shap": lambda m: m.importance("shap"),
    "tuning_table": lambda m: m.tuning_table(),
    "boundary_table": lambda m: m.boundary_table(),
    "params_table": lambda m: m.params_table(),
    "split_summary": lambda m: m.split_summary(),
    "export_code": _export_code_kwargs,
    "residuals_plot": lambda m: m.residuals_plot().to_json(),
    "roc_curve_plot": lambda m: m.roc_curve_plot().to_json(),
    "calibration_plot": lambda m: m.calibration_plot().to_json(),
    "probability_histogram_plot": lambda m: m.probability_histogram_plot().to_json(),
    "importance_plot": lambda m: m.importance_plot().to_json(),
    "plot_learning_curve": lambda m: m.plot_learning_curve().to_json(),
    "plot_oof_distribution": lambda m: m.plot_oof_distribution().to_json(),
    "tuning_plot": lambda m: m.tuning_plot().to_json(),
}


def read(model: Model, surface: str) -> Any:
    try:
        return ("ok", SURFACES[surface](model))
    except LizyMLError as exc:
        return ("raises", exc.code.name)
    except Exception as exc:  # noqa: BLE001 - a raw error is a reading too
        return ("raises", type(exc).__name__)


def same(a: Any, b: Any) -> bool:
    if type(a) is not type(b):
        return False
    if isinstance(a, pd.DataFrame | pd.Series):
        try:
            pd.testing.assert_frame_equal(a, b) if isinstance(
                a, pd.DataFrame
            ) else pd.testing.assert_series_equal(a, b)
        except AssertionError:
            return False
        return True
    if isinstance(a, np.ndarray):
        return a.shape == b.shape and bool(np.array_equal(a, b, equal_nan=True))
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, list | tuple):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, float) and isinstance(b, float):
        return a == b or (np.isnan(a) and np.isnan(b))
    return bool(a == b)


def diff_keys(a: Any, b: Any) -> str:
    if a[0] != b[0]:
        return f"{a[0]}:{a[1] if a[0] == 'raises' else ''} -> {b[0]}:{b[1] if b[0] == 'raises' else ''}"
    if a[0] == "raises":
        return f"raises {a[1]} -> raises {b[1]}"
    x, y = a[1], b[1]
    if isinstance(x, dict):
        keys = sorted(k for k in x.keys() | y.keys() if not same(x.get(k), y.get(k)))
        return "keys " + ", ".join(f"{k}: {x.get(k)!r} -> {y.get(k)!r}" for k in keys)
    if isinstance(x, pd.DataFrame) and x.index.equals(y.index):
        rows = [i for i in x.index if not same(x.loc[[i]], y.loc[[i]])]
        if rows:
            return "rows " + ", ".join(
                f"{i}: {x.loc[i].tolist()} -> {y.loc[i].tolist()}" for i in rows
            )
    if isinstance(x, pd.DataFrame):
        return f"frame shape {x.shape} -> {y.shape}"
    return "value differs"


def run_lifecycle(task: str, name: str, workdir: Path) -> tuple[Model, Model]:
    model = build(task)
    if name == "fit":
        model.fit()
    elif name in ("tune_fit", "tune_fit_reexport"):
        model.tune()
        model.fit()
    elif name == "tune_resume_fit":
        model.tune()
        # expand_boundary defaults to True only for the default space; this
        # space is the caller's, so it is asked for.
        model.tune(resume=True, expand_boundary=True)
        model.fit()
    elif name == "fit_tune":
        model.fit()
        model.tune()
    first = workdir / f"{task}-{name}-1"
    model.export(first)
    loaded = Model.load(first)
    if name == "tune_fit_reexport":
        second = workdir / f"{task}-{name}-2"
        loaded.export(second)
        loaded = Model.load(second)
    return model, loaded


def main() -> int:
    total = 0
    differing: list[str] = []
    raised_both: dict[str, int] = {}
    exercised: dict[str, int] = {}
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        for task in DATA:
            for name in LIFECYCLES:
                fitted, loaded = run_lifecycle(task, name, workdir)
                for surface in SURFACES:
                    total += 1
                    before, after = read(fitted, surface), read(loaded, surface)
                    if before[0] == after[0] == "raises" and before[1] == after[1]:
                        raised_both[surface] = raised_both.get(surface, 0) + 1
                    if before[0] == "ok":
                        exercised[surface] = exercised.get(surface, 0) + 1
                    if same(before, after):
                        continue
                    differing.append(
                        f"{task:<17} {name:<18} {surface:<27} "
                        f"{diff_keys(before, after)}"
                    )
    for line in differing:
        print(line)
    print()
    print(
        f"cells: {total} ({len(DATA)} configurations x {len(LIFECYCLES)} lifecycles x "
        f"{len(SURFACES)} surfaces); differing: {len(differing)}"
    )
    print("raised identically before and after (no reading to lose in that cell):")
    for surface, n in sorted(raised_both.items()):
        print(f"  {surface}: {n}")
    print("cells where the fitted model returned a reading (exercised):")
    for surface in SURFACES:
        print(f"  {surface}: {exercised.get(surface, 0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
