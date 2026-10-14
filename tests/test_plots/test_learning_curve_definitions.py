"""The learning curve's LightGBM definitions are annotated where they differ (H-0125).

#354: ``FitResult.history`` is LightGBM's evaluation, under LightGBM's names and
definitions, and two of its names are also LizyML metrics with another
definition (``huber``: LightGBM's threshold is ``alpha``; ``mape``: LightGBM's
ratio over ``max(1, |y|)``). ``plot_learning_curve`` titles those subplots
with LightGBM's definition, from one table,
:data:`lizyml.plots.learning_curve.LIGHTGBM_CURVE_DEFINITIONS`.

Criterion 9 pins that table to reality instead of to a copy of it: real
LightGBM records every accepted native metric of each task, the recorded names
that are also ``MetricRegistry`` names of that task are evaluated by LizyML on
the same predictions, and the names whose values differ must be exactly the
table's keys. A key that does not differ, or that is not such a shared name,
fails as surely as a differing name the table lacks; the last two tests are the
negative controls of that comparison. Criterion 10 checks the titles and that
the ``metrics=`` filter still matches the bare name.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from lizyml import Model
from lizyml.core.types.fit_result import FitResult
from lizyml.estimators.lgbm.adapter import LGBMAdapter
from lizyml.estimators.lgbm.metric_bridge import _LGBM_NATIVE_METRICS
from lizyml.evaluation.evaluator import _pred_for_metric
from lizyml.metrics.registry import _TASK_METRICS, get_metric
from lizyml.plots.learning_curve import LIGHTGBM_CURVE_DEFINITIONS, plot_learning_curve
from tests._helpers import make_config, make_regression_df

_TASKS = ("regression", "binary", "multiclass")
_N_TRAIN = 300
_N_VALID = 150
_ROUNDS = 15
# LightGBM stores labels as float32, so a metric both define identically still
# differs by ~1e-9 relative (rmse on this data: 3e-9). The definitions that
# differ do so by far more (huber here: 8e-2; mape: a factor of ~200).
_REL_TOL = 1e-6


def _data(task: str) -> tuple[pd.DataFrame, pd.Series]:
    """Seeded data on which the differing definitions show.

    The regression target is positive (``gamma``, ``poisson``, ``tweedie`` and
    both ``mape`` need it) and its errors are large enough that many exceed
    LightGBM's ``alpha`` 0.9, where the two ``huber`` definitions part.
    """
    rng = np.random.default_rng(0)
    n = _N_TRAIN + _N_VALID
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=["a", "b", "c"])
    noise = rng.normal(scale=0.7, size=n)
    if task == "regression":
        y = np.exp(0.8 * X["a"].to_numpy() + noise) * 2.0
    elif task == "binary":
        y = (X["a"].to_numpy() + noise > 0).astype(int)
    else:
        y = np.digitize(X["a"].to_numpy() + noise, [-0.5, 0.5])
    return X, pd.Series(y)


def _comparison(task: str) -> tuple[set[str], dict[str, tuple[float, float]]]:
    """Record every accepted native metric; compare the shared names.

    Returns the recorded names and, for each recorded name that is also a
    ``MetricRegistry`` name of the task, ``(LightGBM value, LizyML value)`` on
    the same predictions: the validation rows, predicted by the booster after
    its last recorded iteration (no early stopping, so ``predict`` uses every
    tree).
    """
    X, y = _data(task)
    adapter = LGBMAdapter(
        task=task,  # type: ignore[arg-type]
        params={"metric": sorted(_LGBM_NATIVE_METRICS[task]), "n_estimators": _ROUNDS},
        num_class=3 if task == "multiclass" else None,
        early_stopping_rounds=None,
    )
    X_train, y_train = X.iloc[:_N_TRAIN], y.iloc[:_N_TRAIN]
    X_valid, y_valid = X.iloc[_N_TRAIN:], y.iloc[_N_TRAIN:]
    adapter.fit(X_train, y_train, X_valid, y_valid)
    recorded = adapter.eval_results["valid_0"]
    assert all(len(values) == _ROUNDS for values in recorded.values())

    booster = adapter.get_native_model()
    raw = np.asarray(booster.predict(X_valid), dtype=np.float64)
    shared = set(recorded) & _TASK_METRICS[task]  # type: ignore[index]
    values: dict[str, tuple[float, float]] = {}
    for name in sorted(shared):
        metric = get_metric(name)
        ours = metric(y_valid.to_numpy(), _pred_for_metric(metric, raw, task))  # type: ignore[arg-type]
        values[name] = (float(recorded[name][-1]), float(ours))
    return set(recorded), values


@pytest.fixture(scope="module")
def comparisons() -> dict[str, tuple[set[str], dict[str, tuple[float, float]]]]:
    return {task: _comparison(task) for task in _TASKS}


def _check_table(
    table: Mapping[str, str],
    comparisons: Mapping[str, tuple[set[str], dict[str, tuple[float, float]]]],
) -> None:
    """The names whose two values differ are exactly the table's keys."""
    shared: set[str] = set()
    differing: set[str] = set()
    for _recorded, values in comparisons.values():
        shared |= set(values)
        differing |= {
            name
            for name, (lgbm, ours) in values.items()
            if not math.isclose(lgbm, ours, rel_tol=_REL_TOL)
        }
    keys = set(table)
    assert not keys - shared, (
        f"annotated names no task records under a LizyML name: {sorted(keys - shared)}"
    )
    assert keys == differing, (
        f"the learning curve and FitResult.metrics differ on {sorted(differing)}, "
        f"but the annotation table names {sorted(keys)}"
    )


def _recorded_alone(task: str, spelling: str) -> set[str]:
    """The names LightGBM records when *spelling* is the only metric."""
    X, y = _data(task)
    adapter = LGBMAdapter(
        task=task,  # type: ignore[arg-type]
        params={"metric": [spelling], "n_estimators": 2},
        num_class=3 if task == "multiclass" else None,
        early_stopping_rounds=None,
    )
    adapter.fit(
        X.iloc[:_N_TRAIN], y.iloc[:_N_TRAIN], X.iloc[_N_TRAIN:], y.iloc[_N_TRAIN:]
    )
    return set(adapter.eval_results["valid_0"])


@pytest.mark.parametrize("task", _TASKS)
def test_every_native_metric_was_recorded(
    comparisons: dict[str, tuple[set[str], dict[str, tuple[float, float]]]],
    task: str,
) -> None:
    """Each accepted native spelling is recorded, and the comparison saw it.

    LightGBM records a spelling under its canonical name (``mae`` as ``l1``),
    so each spelling of the task's native table is trained alone and must be
    recorded under exactly one name; the union of those names must equal what
    the comparison's run recorded. A spelling LightGBM ignored, or a name the
    comparison's run lost, fails here instead of passing unseen.
    """
    per_spelling = {
        spelling: _recorded_alone(task, spelling)
        for spelling in sorted(_LGBM_NATIVE_METRICS[task])
    }
    unrecorded = {s: names for s, names in per_spelling.items() if len(names) != 1}
    assert not unrecorded, (
        f"{task}: spellings not recorded under one name: {unrecorded}"
    )
    expected = set().union(*per_spelling.values())
    assert comparisons[task][0] == expected, task


def test_the_shared_names_are_known(
    comparisons: dict[str, tuple[set[str], dict[str, tuple[float, float]]]],
) -> None:
    """The recorded names that are also LizyML metric names, per task."""
    assert set(comparisons["regression"][1]) == {"huber", "mape", "rmse"}
    assert set(comparisons["binary"][1]) == {"auc"}
    assert comparisons["multiclass"][1] == {}


def test_annotation_table_is_the_set_of_differing_names(
    comparisons: dict[str, tuple[set[str], dict[str, tuple[float, float]]]],
) -> None:
    """Criterion 9: the table's keys equal the shared names whose values differ."""
    _check_table(LIGHTGBM_CURVE_DEFINITIONS, comparisons)


def test_a_table_missing_a_differing_name_fails(
    comparisons: dict[str, tuple[set[str], dict[str, tuple[float, float]]]],
) -> None:
    """Negative control: dropping ``mape`` from the table is caught."""
    table = {k: v for k, v in LIGHTGBM_CURVE_DEFINITIONS.items() if k != "mape"}
    with pytest.raises(AssertionError, match="mape"):
        _check_table(table, comparisons)


@pytest.mark.parametrize(
    "extra",
    [
        pytest.param("rmse", id="shared-but-equal"),
        pytest.param("l1", id="not-a-shared-name"),
    ],
)
def test_a_table_with_a_name_that_does_not_differ_fails(
    comparisons: dict[str, tuple[set[str], dict[str, tuple[float, float]]]],
    extra: str,
) -> None:
    """Negative control: a key that does not differ, or is not shared, fails."""
    table = {**LIGHTGBM_CURVE_DEFINITIONS, extra: "anything"}
    with pytest.raises(AssertionError, match=extra):
        _check_table(table, comparisons)


# ---------------------------------------------------------------------------
# Criterion 10: the subplot titles
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fit_result() -> FitResult:
    """A fitted regression whose history holds huber, l1, mape and rmse."""
    cfg = make_config("regression", n_estimators=20)
    cfg["model"]["params"]["metric"] = ["huber", "mae", "mape", "rmse"]
    model = Model(cfg)
    return model.fit(data=make_regression_df(n=200))


def _titles(fig: go.Figure) -> list[str]:
    return [ann.text for ann in fig.layout.annotations if getattr(ann, "text", None)]


def test_subplot_titles_carry_lightgbm_definitions(fit_result: FitResult) -> None:
    titles = _titles(plot_learning_curve(fit_result))
    assert titles == [
        "valid_0/huber (LightGBM: threshold = alpha)",
        "valid_0/l1",
        "valid_0/mape (LightGBM: |y-p| / max(1,|y|))",
        "valid_0/rmse",
    ]


def test_every_table_key_is_annotated_from_the_table(fit_result: FitResult) -> None:
    """The title is built from the table, so the table is the only copy."""
    titles = _titles(plot_learning_curve(fit_result))
    for name, definition in LIGHTGBM_CURVE_DEFINITIONS.items():
        assert f"valid_0/{name} (LightGBM: {definition})" in titles


def test_metrics_filter_matches_the_bare_name(fit_result: FitResult) -> None:
    fig = plot_learning_curve(fit_result, metrics=["huber"])
    assert _titles(fig) == ["valid_0/huber (LightGBM: threshold = alpha)"]
    # The legend (one trace per fold) and the axis titles are unchanged.
    n_folds = len(fit_result.history)
    assert [trace.name for trace in fig.data] == [f"fold {i}" for i in range(n_folds)]
    assert fig.layout.yaxis.title.text == "Loss"
    assert fig.layout.xaxis.title.text == "Iteration"


def test_annotated_title_does_not_match_as_a_filter(fit_result: FitResult) -> None:
    """The filter takes the metric name, not the displayed title."""
    from lizyml.core.exceptions import ErrorCode, LizyMLError

    with pytest.raises(LizyMLError) as exc_info:
        plot_learning_curve(fit_result, metrics=["huber (LightGBM: threshold = alpha)"])
    assert exc_info.value.code == ErrorCode.CONFIG_INVALID
    assert "huber" in exc_info.value.context["available_metrics"]
