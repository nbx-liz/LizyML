"""The export -> load census: every public reporting reading, before and after.

One definition, two readers: ``test_reporting_surfaces_survive_load.py`` asserts
on it, and ``docs/audits/2026-09-defect-discovery/instruments/pr8b_load_census.py``
prints it (H-0109, #281). Keeping the surfaces in one place is the point: a
census copied into the instrument would drift from the one the test enforces.

**Population.** ``INVENTORY`` classifies every public name on ``Model``: either
the surfaces it is read through, or why it is not a reporting surface. A test
asserts the inventory equals ``dir(Model)``'s public names, so a new public
method has to be classified before the suite passes.

**Bound.** Each surface is called with its defaults, except ``importance``,
which is read for each of its three kinds. Non-default arguments of
``confusion_matrix(threshold=)``, ``importance_plot(kind=, top_n=)``,
``plot_learning_curve(metrics=)``, ``residuals_plot(kind=)``,
``evaluate(metrics=)`` and ``predict(return_shap=)`` are not exercised.

**Projections.** Three readings compare a projection, not the whole object:
``predict`` compares ``PredictionResult.pred``; ``fit_result`` compares every
``FitResult`` field except ``models`` / ``pipeline_state`` / ``calibrator`` /
``target_encoder``, which are compared by type name (their behaviour is what
``predict`` and the other surfaces read); ``export_code`` compares the arguments
passed to ``generate_code``, with its live-object arguments compared by type.
"""

from __future__ import annotations

import dataclasses
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import pandas as pd

from lizyml.core.exceptions import LizyMLError
from lizyml.core.model import Model
from tests._helpers import (
    make_binary_df,
    make_config,
    make_multiclass_df,
    make_regression_df,
)

CONFIG_RATIO = 0.2
TUNED_RATIO = 0.45

#: The tolerance for ``importance("gain")`` across load. LightGBM writes each
#: split's gain with six significant digits and reads it back into a binary32
#: ``float``. The decimal rounding moves a split gain g by at most 5e-6 |g|;
#: the binary32 read (under round-to-nearest) of a value d in binary32's finite
#: range moves it by at most 2**-24 |d| + 2**-150 (the absolute term covers the
#: inputs where the usual relative bound 2**-24 does not hold: throughout the
#: subnormal range, and just under 2**-126 where values round up into the normal
#: range). So the
#: loaded gain moves by at most C |g| + 2**-150, C = (1 + 5e-6)(1 + 2**-24) - 1,
#: approximately 5.0596e-6. A feature's gain is a sum of n non-negative split
#: gains G, so it moves by at most C G + n 2**-150, and n 2**-150 < 2**-126 for
#: fewer than 2**24 splits. ``pr8b_gain_precision.py``
#: sweeps every decimal exponent of positive binary32, subnormals included, and
#: finds every split gain of the models it trains, in three tasks, positive.
GAIN_RTOL = 5.1e-6
GAIN_ATOL = 2.0**-126

SPACE: dict[str, Any] = {
    "early_stopping_rounds": {
        "type": "categorical",
        "choices": [2],
        "category": "training",
    },
    "validation_ratio": {
        "type": "categorical",
        "choices": [TUNED_RATIO],
        "category": "training",
    },
}

#: Configurations, not only tasks: ``binary_calibrated`` is the one in which the
#: calibration surfaces have a reading at all.
CONFIGURATIONS: dict[str, Callable[..., pd.DataFrame]] = {
    "regression": make_regression_df,
    "binary": make_binary_df,
    "binary_calibrated": make_binary_df,
    "multiclass": make_multiclass_df,
}
#: ``tune_resume_fit`` is the only order that produces a boundary report, so the
#: only one in which ``boundary_table`` has a reading to lose.
TUNED_LIFECYCLES = ("tune_fit", "fit_tune", "tune_fit_reexport", "tune_resume_fit")
LIFECYCLES = ("fit", *TUNED_LIFECYCLES)

#: Live objects in ``generate_code``'s arguments and in ``FitResult``: compared
#: by type, because their behaviour is what the other surfaces read.
OBJECT_KWARGS = frozenset(
    {"model_adapter", "pipeline_state", "calibrator", "output_dir"}
)
OBJECT_FIELDS = frozenset({"models", "pipeline_state", "calibrator", "target_encoder"})


def _export_code_kwargs(model: Model, X: pd.DataFrame) -> dict[str, Any]:
    with mock.patch("lizyml.codegen.generator.generate_code") as generate:
        model.export_code("not-written")
    return {
        key: (type(value).__name__ if key in OBJECT_KWARGS else value)
        for key, value in generate.call_args.kwargs.items()
    }


def _fit_result(model: Model, X: pd.DataFrame) -> dict[str, Any]:
    result = model.fit_result
    out: dict[str, Any] = {}
    for field in dataclasses.fields(result):
        value = getattr(result, field.name)
        if field.name in OBJECT_FIELDS:
            out[field.name] = type(value).__name__
        elif dataclasses.is_dataclass(value) and not isinstance(value, type):
            out[field.name] = dataclasses.asdict(value)
        else:
            out[field.name] = value
    return out


SURFACES: dict[str, Callable[[Model, pd.DataFrame], Any]] = {
    "evaluate": lambda m, X: m.evaluate(),
    "evaluate_table": lambda m, X: m.evaluate_table(),
    "fit_result": _fit_result,
    "predict": lambda m, X: m.predict(X).pred,
    "residuals": lambda m, X: m.residuals(),
    "confusion_matrix": lambda m, X: m.confusion_matrix(),
    "importance_split": lambda m, X: m.importance("split"),
    "importance_gain": lambda m, X: m.importance("gain"),
    "importance_shap": lambda m, X: m.importance("shap"),
    "tuning_table": lambda m, X: m.tuning_table(),
    "boundary_table": lambda m, X: m.boundary_table(),
    "params_table": lambda m, X: m.params_table(),
    "split_summary": lambda m, X: m.split_summary(),
    "export_code": _export_code_kwargs,
    "residuals_plot": lambda m, X: m.residuals_plot().to_json(),
    "roc_curve_plot": lambda m, X: m.roc_curve_plot().to_json(),
    "calibration_plot": lambda m, X: m.calibration_plot().to_json(),
    "probability_histogram_plot": lambda m, X: m.probability_histogram_plot().to_json(),
    "importance_plot": lambda m, X: m.importance_plot().to_json(),
    "plot_learning_curve": lambda m, X: m.plot_learning_curve().to_json(),
    "plot_oof_distribution": lambda m, X: m.plot_oof_distribution().to_json(),
    "tuning_plot": lambda m, X: m.tuning_plot().to_json(),
}

#: Every public name on ``Model``: a tuple of the surfaces it is read through, or
#: the reason it is not a reporting surface.
INVENTORY: dict[str, tuple[str, ...] | str] = {
    "boundary_table": ("boundary_table",),
    "calibration_plot": ("calibration_plot",),
    "confusion_matrix": ("confusion_matrix",),
    "evaluate": ("evaluate",),
    "evaluate_table": ("evaluate_table",),
    "export": "writes the artifact the census reads",
    "export_code": ("export_code",),
    "fit": "trains; it is a lifecycle step here, not a reading",
    "fit_result": ("fit_result",),
    "importance": ("importance_split", "importance_gain", "importance_shap"),
    "importance_plot": ("importance_plot",),
    "load": "reads the artifact; every reading after it is a census cell",
    "params_table": ("params_table",),
    "plot_learning_curve": ("plot_learning_curve",),
    "plot_oof_distribution": ("plot_oof_distribution",),
    "predict": ("predict",),
    "probability_histogram_plot": ("probability_histogram_plot",),
    "residuals": ("residuals",),
    "residuals_plot": ("residuals_plot",),
    "roc_curve_plot": ("roc_curve_plot",),
    "split_summary": ("split_summary",),
    "tune": "searches; it is a lifecycle step here, not a reading",
    "tuning_plot": ("tuning_plot",),
    "tuning_table": ("tuning_table",),
}

#: #315: the only cells allowed to differ. H-0086 persists the tuning result's
#: ``best_*`` overlay and not its trials, rounds or boundary report.
DECLARED: frozenset[tuple[str, str, str]] = frozenset(
    {
        (config, lifecycle, surface)
        for config in CONFIGURATIONS
        for lifecycle in TUNED_LIFECYCLES
        for surface in ("tuning_table", "tuning_plot")
    }
    | {(config, "tune_resume_fit", "boundary_table") for config in CONFIGURATIONS}
)


def build(config: str) -> tuple[Model, pd.DataFrame]:
    task = "binary" if config == "binary_calibrated" else config
    cfg = make_config(
        task, n_estimators=10, n_splits=2, num_threads=1, tuning_n_trials=1
    )
    cfg["training"]["early_stopping"] = {
        "enabled": True,
        "rounds": 7,
        "validation_ratio": CONFIG_RATIO,
    }
    cfg["tuning"]["optuna"]["space"] = dict(SPACE)
    if config == "binary_calibrated":
        cfg["calibration"] = {"method": "platt"}
    df = CONFIGURATIONS[config](n=200)
    return Model(cfg, data=df), df.drop(columns=["target"])


def run(
    config: str, lifecycle: str, workdir: Path
) -> tuple[Model, Model, pd.DataFrame]:
    model, X = build(config)
    if lifecycle == "fit":
        model.fit()
    elif lifecycle in ("tune_fit", "tune_fit_reexport"):
        model.tune()
        model.fit()
    elif lifecycle == "fit_tune":
        model.fit()
        model.tune()
    elif lifecycle == "tune_resume_fit":
        model.tune()
        # expand_boundary defaults to True only for the default space.
        model.tune(resume=True, expand_boundary=True)
        model.fit()
    loaded = Model.load(model.export(workdir / f"{config}-{lifecycle}-1"))
    if lifecycle == "tune_fit_reexport":
        loaded = Model.load(loaded.export(workdir / f"{config}-{lifecycle}-2"))
    return model, loaded, X


def read(model: Model, surface: str, X: pd.DataFrame) -> tuple[str, Any]:
    try:
        return ("ok", SURFACES[surface](model, X))
    except LizyMLError as exc:
        return ("raises", exc.code.name)


def same(a: Any, b: Any) -> bool:
    if type(a) is not type(b):
        return False
    if isinstance(a, pd.DataFrame | pd.Series):
        return bool(a.equals(b)) and (
            not isinstance(a, pd.DataFrame) or a.columns.equals(b.columns)
        )
    if isinstance(a, np.ndarray):
        return a.shape == b.shape and bool(np.array_equal(a, b, equal_nan=True))
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, list | tuple):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, float):
        return a == b or (np.isnan(a) and np.isnan(b))
    return bool(a == b)


def gain_within_bound(a: tuple[str, Any], b: tuple[str, Any]) -> bool:
    if a[0] != "ok" or b[0] != "ok" or a[1].keys() != b[1].keys():
        return False
    return all(
        np.isclose(b[1][k], a[1][k], rtol=GAIN_RTOL, atol=GAIN_ATOL) for k in a[1]
    )


def agrees(surface: str, before: tuple[str, Any], after: tuple[str, Any]) -> bool:
    if surface == "importance_gain":
        return same(before, after) or gain_within_bound(before, after)
    return same(before, after)


def collect(workdir: Path) -> list[dict[str, Any]]:
    """Every (configuration, lifecycle, surface) reading, before and after load."""
    cells: list[dict[str, Any]] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for config in CONFIGURATIONS:
            for lifecycle in LIFECYCLES:
                fitted, loaded, X = run(config, lifecycle, workdir)
                for surface in SURFACES:
                    cells.append(
                        {
                            "cell": (config, lifecycle, surface),
                            "before": read(fitted, surface, X),
                            "after": read(loaded, surface, X),
                        }
                    )
    return cells
