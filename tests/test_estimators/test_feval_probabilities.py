"""The LightGBM feval reports the metric of the predictions LightGBM produced (#306).

For built-in objectives LightGBM 4 hands a feval the **probabilities** --
binary 1-D, multiclass and multiclassova 2-D ``(n, num_class)``. The bridge used
to assume raw scores and applied ``sigmoid`` / ``softmax`` again: binary
``accuracy`` and ``f1`` became constant (every row positive), early stopping
stopped every fold at iteration 1, and probability metrics were distorted.

The tests below run the real ``lgb.train``. The expected value is computed from
what LightGBM actually passed, with the evaluator's own prediction rule
(``evaluation.evaluator._pred_for_metric``), so neither side is a hand-written
fixture of the same assumption -- which is how the old tests missed this.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from lizyml.core.model import Model
from lizyml.estimators.lgbm.defaults import TASK_COMPATIBLE_OBJECTIVES
from lizyml.estimators.lgbm.metric_bridge import _FEVAL_METRICS, _build_feval
from lizyml.evaluation.evaluator import _pred_for_metric
from lizyml.metrics.registry import get_metric
from tests._helpers import make_config

#: (task, LightGBM objective, extra params): every objective LizyML accepts for
#: a task, read from its own table rather than listed here (review round 1 found
#: a hand-written list that omitted ``cross_entropy`` / ``cross_entropy_lambda``).
OBJECTIVES: list[tuple[str, str, dict[str, Any]]] = [
    (task, objective, {"num_class": 3} if task == "multiclass" else {})
    for task, objectives in sorted(TASK_COMPATIBLE_OBJECTIVES.items())
    for objective in sorted(objectives)
]

CELLS = [
    pytest.param(task, objective, extra, name, id=f"{objective}-{name}")
    for task, objective, extra in OBJECTIVES
    for name in sorted(_FEVAL_METRICS[task])  # type: ignore[index]
]


def test_the_population_is_every_feval_metric() -> None:
    names = {(c.values[0], c.values[3]) for c in CELLS}
    expected = {
        (task, name) for task, names_ in _FEVAL_METRICS.items() for name in names_
    }
    assert {(t, n) for t, n in names} == expected


def _data(task: str, n: int = 400) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, 4))
    signal = X[:, 0] + 0.5 * X[:, 1] + rng.normal(scale=0.7, size=n)
    if task == "regression":
        return X, np.exp(0.3 * signal)  # positive, for rmsle
    if task == "binary":
        return X, (signal > 0.3).astype(int)
    return X, np.digitize(signal, [-0.5, 0.6])


@pytest.mark.parametrize(("task", "objective", "extra", "name"), CELLS)
def test_feval_value_is_the_metric_of_lightgbm_predictions(
    task: str, objective: str, extra: dict[str, Any], name: str
) -> None:
    """Same value as the evaluator's rule on LightGBM's output -- or, where that
    rule itself fails (an objective whose output is not a probability, such as
    ``cross_entropy_lambda`` above 1, fed to a probability metric), the same
    failure: the learning curve must not report a number ``FitResult.metrics``
    could not compute (H-0105 decision 1)."""
    metric = get_metric(name)
    feval = _build_feval(metric, task, num_class=extra.get("num_class"))  # type: ignore[arg-type]
    pairs: list[tuple[float, float]] = []

    def checked(preds: np.ndarray, data: lgb.Dataset) -> tuple[str, float, bool]:
        y_true = np.asarray(data.get_label())
        try:
            expected = metric(y_true, _pred_for_metric(metric, np.asarray(preds), task))  # type: ignore[arg-type]
        except Exception as exc:  # noqa: BLE001 -- the outcome is compared, not swallowed
            with pytest.raises(type(exc)):
                feval(preds, data)
            pairs.append((np.nan, np.nan))
            return name, 0.0, metric.greater_is_better
        out = feval(preds, data)
        pairs.append((out[1], expected))
        return out

    X, y = _data(task)
    ds = lgb.Dataset(X, y)
    lgb.train(
        {"objective": objective, "verbosity": -1, "metric": "None", **extra},
        ds,
        num_boost_round=15,
        valid_sets=[ds],
        feval=checked,
    )
    assert pairs, "the feval was never called"
    got, want = np.array(pairs).T
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12, equal_nan=True)


def _binary_df(n: int = 2000) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, 5))
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
    df["target"] = (
        X[:, 0] + 0.5 * X[:, 1] + rng.normal(scale=0.8, size=n) > 0.3
    ).astype(int)
    return df


@pytest.mark.parametrize("metric", ["accuracy", "f1"])
def test_label_metric_does_not_stop_training_at_iteration_one(metric: str) -> None:
    """#306's observable: with the double sigmoid every row was positive, the
    metric never moved, and early stopping kept one tree in every fold."""
    cfg = make_config("binary", n_estimators=300, n_splits=3)
    cfg["model"]["params"]["learning_rate"] = 0.05
    cfg["model"]["params"]["metric"] = metric
    cfg["training"]["early_stopping"] = {"enabled": True, "rounds": 20}
    fit_result = Model(cfg).fit(data=_binary_df())

    best = [h["best_iteration"] for h in fit_result.history]
    assert min(best) > 5, f"early stopping kept {best} trees per fold"
    curve = fit_result.history[0]["eval_history"]["valid_0"][metric]
    assert len(set(np.round(curve, 12))) > 1, "the feval metric never changed"


def _load(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        f"gen_{name}_{path.name}", path / f"{name}.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("task", "metric", "n_classes"),
    [
        ("binary", "accuracy", None),
        ("binary", "brier", None),
        ("multiclass", "brier", 3),
    ],
)
def test_generated_feval_matches_the_metric(
    task: str, metric: str, n_classes: int | None, tmp_path: Path
) -> None:
    """The generated train.py carried the same double transform."""
    rng = np.random.default_rng(1)
    n = 300
    X = rng.normal(size=(n, 3))
    df = pd.DataFrame(X, columns=["a", "b", "c"])
    signal = X[:, 0] + rng.normal(scale=0.5, size=n)
    df["target"] = (
        (signal > 0).astype(int)
        if task == "binary"
        else np.digitize(signal, [-0.5, 0.5])
    )
    cfg = make_config(task, n_estimators=20, n_splits=2)
    cfg["model"]["params"]["metric"] = metric
    model = Model(cfg)
    model.fit(data=df)
    model.export_code(tmp_path / "gen")
    train = _load(tmp_path / "gen", "train")

    fevals = train.build_feval_from_config()
    assert len(fevals) == 1
    probs = (
        rng.random(n)
        if task == "binary"
        else (lambda a: a / a.sum(axis=1, keepdims=True))(rng.random((n, n_classes)))
    )
    ds = lgb.Dataset(X, df["target"].to_numpy(), free_raw_data=False)
    ds.construct()
    _, value, _ = fevals[0](probs, ds)
    m = get_metric(metric)
    expected = m(df["target"].to_numpy(), _pred_for_metric(m, probs, task))  # type: ignore[arg-type]
    assert value == pytest.approx(expected, rel=1e-12, abs=1e-12)
