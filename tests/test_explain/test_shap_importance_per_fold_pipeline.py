"""SHAP importance explains each fold model on rows its own pipeline encoded (H-0114).

Each CV fold's model is trained, and predicts its OOF rows, on data transformed
by that fold's feature pipeline. ``importance(kind="shap")`` must explain fold k
on fold k's validation rows transformed the same way, not with the last fold's
pipeline (#303).

The fold pipelines differ only for a string column that ``auto_categorical:
false`` leaves to the pipeline: each fold then learns its own category set. The
data below makes the category set change over time, and LightGBM's categorical
defaults are relaxed so that the fold models actually split on it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from lizyml import Model
from tests._helpers import make_config

pytest.importorskip("shap")

from lizyml.explain.shap_explainer import (  # noqa: E402
    compute_shap_importance,
    compute_shap_values,
)

_TASKS = ("regression", "binary", "multiclass")


def _with_task_target(df: pd.DataFrame, score: pd.Series, task: str) -> pd.DataFrame:
    if task == "regression":
        df["target"] = score
    elif task == "binary":
        df["target"] = (score > score.median()).astype(int)
    else:
        df["target"] = pd.qcut(score, 3, labels=False).astype(int)
    return df


def _drift_df(task: str = "regression", n: int = 600) -> pd.DataFrame:
    """Value "b" appears only in rows 150-300: early folds do not know it."""
    rng = np.random.default_rng(0)
    cat = rng.choice(["a", "c", "d"], n).astype(object)
    cat[150:300][rng.random(150) < 0.4] = "b"
    effect = {"a": 0.0, "b": 3.0, "c": -2.0, "d": 1.0}
    df = pd.DataFrame({"t": np.arange(n), "num": rng.normal(size=n), "cat": cat})
    score = df["cat"].map(effect) + df["num"] + rng.normal(scale=0.1, size=n)
    return _with_task_target(df, score, task)


def _sliding_df(task: str = "regression", n: int = 600) -> pd.DataFrame:
    """Value "e" appears only in rows 100-270.

    With four folds and ``train_size_max: 200`` the last fold trains on rows
    280-479, so it does not know "e", while fold 1 (train 40-239, valid
    240-359) knows it and holds it in its validation rows. Every fold that holds
    "e" in its validation rows also knows it, so ``unseen_policy: "error"``
    lets ``fit`` succeed.

    "e" is frequent and its effect is the largest, so the fold models split on
    "e" itself. With a rarer, milder "e" (40 %, effect -4) the boosters split
    only on "c" and "d", sending "e" down the same branch as a missing value,
    and the ``nan`` comparison could not tell the two pipelines apart.
    """
    rng = np.random.default_rng(1)
    cat = rng.choice(["a", "c", "d"], n).astype(object)
    cat[100:270][rng.random(170) < 0.6] = "e"
    effect = {"a": 0.0, "c": -2.0, "d": 1.0, "e": 6.0}
    df = pd.DataFrame({"t": np.arange(n), "num": rng.normal(size=n), "cat": cat})
    score = df["cat"].map(effect) + df["num"] + rng.normal(scale=0.1, size=n)
    return _with_task_target(df, score, task)


def _config(
    task: str,
    policy: str,
    *,
    sliding: bool = False,
    features: dict[str, Any] | None = None,
) -> dict[str, Any]:
    raw = make_config(
        task,
        n_estimators=40,
        n_splits=4,
        split_method="time_series",
        time_col="t",
        split_overrides={"train_size_max": 200} if sliding else None,
        min_data_per_group=5,
        cat_smooth=1.0,
        cat_l2=1.0,
    )
    raw["features"] = (
        features
        if features is not None
        else {"unseen_policy": policy, "auto_categorical": False}
    )
    return raw


def _fit(raw: dict[str, Any], df: pd.DataFrame) -> Model:
    model = Model(raw)
    model.fit(data=df)
    return model


def _fold_own_pipelines(model: Model, policy: str) -> list[Any]:
    """Refit each fold's pipeline on its training rows, as CVTrainer does."""
    state = model._get_fit_state()
    assert state.X is not None
    y = pd.Series(np.asarray(state.y))
    factory = state.provider.build_pipeline_factory(unseen_policy=policy)  # type: ignore[arg-type]
    pipelines = []
    for train_idx, _ in state.fit_result.splits.outer:
        pipeline = factory()
        pipeline.fit(
            state.X.iloc[train_idx].reset_index(drop=True),
            y.iloc[train_idx].reset_index(drop=True),
        )
        pipelines.append(pipeline)
    return pipelines


def _importance_from(
    model: Model, encoded_valid: list[pd.DataFrame]
) -> dict[str, float]:
    """fold-mean of mean(|SHAP|) over the given encoded validation rows."""
    state = model._get_fit_state()
    task = state.cfg.task
    names = state.fit_result.feature_names
    agg = np.zeros(len(names))
    for fold_model, X_valid in zip(state.fit_result.models, encoded_valid, strict=True):
        shap_vals = compute_shap_values(fold_model, X_valid, task)
        agg += np.mean(np.abs(shap_vals), axis=0)
    agg /= len(encoded_valid)
    return {name: float(agg[i]) for i, name in enumerate(names)}


def _replay_fold_own(model: Model, policy: str) -> dict[str, float]:
    """Independent reference: fold k's valid rows encoded by fold k's own pipeline."""
    state = model._get_fit_state()
    assert state.X is not None
    encoded = [
        pipeline.transform(state.X.iloc[valid_idx].reset_index(drop=True))
        for pipeline, (_, valid_idx) in zip(
            _fold_own_pipelines(model, policy),
            state.fit_result.splits.outer,
            strict=True,
        )
    ]
    return _importance_from(model, encoded)


def _replay_last_fold(model: Model) -> dict[str, float]:
    """The pre-H-0114 computation: every row encoded by the last fold's pipeline."""
    state = model._get_fit_state()
    assert state.X is not None
    pipeline = state.provider.build_pipeline_factory()()
    pipeline.load_state(state.fit_result.pipeline_state)
    X_t, _ = pipeline.transform_with_warnings(state.X)
    encoded = [X_t.iloc[valid_idx] for _, valid_idx in state.fit_result.splits.outer]
    return _importance_from(model, encoded)


def _fold_categories(pipeline: Any) -> list[str]:
    return list(pipeline.get_state()["encoder"]["categories"].get("cat", []))


def _affected_folds(model: Model, policy: str) -> list[int]:
    """Folds whose validation rows hold a value only one of the two pipelines knows."""
    state = model._get_fit_state()
    assert state.X is not None
    pipelines = _fold_own_pipelines(model, policy)
    last = set(_fold_categories(pipelines[-1]))
    affected = []
    for k, (pipeline, (_, valid_idx)) in enumerate(
        zip(pipelines, state.fit_result.splits.outer, strict=True)
    ):
        differing = set(_fold_categories(pipeline)) ^ last
        if differing & set(state.X.iloc[valid_idx]["cat"]):
            affected.append(k)
    return affected


def _assert_affected_folds_split_on_cat(model: Model, policy: str) -> None:
    affected = _affected_folds(model, policy)
    assert affected, "the data no longer makes any fold differ from the last one"
    models = model._get_fit_state().fit_result.models
    for k in affected:
        assert models[k].importance(kind="split").get("cat", 0) > 0, k


def _assert_matches(actual: dict[str, float], expected: dict[str, float]) -> None:
    assert actual.keys() == expected.keys()
    for name, value in expected.items():
        assert actual[name] == pytest.approx(value, rel=1e-9, abs=1e-12), name


def _assert_differs(actual: dict[str, float], other: dict[str, float]) -> None:
    assert abs(actual["cat"] - other["cat"]) > 1e-6 * max(abs(other["cat"]), 1e-12)


# ---------------------------------------------------------------------------
# Acceptance 1: a value the fold does not know but the last fold does
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("task", _TASKS)
def test_fold_unknown_value_is_explained_as_the_fold_saw_it(task: str) -> None:
    model = _fit(_config(task, "mode"), _drift_df(task))
    _assert_affected_folds_split_on_cat(model, "mode")

    actual = model.importance(kind="shap")

    _assert_matches(actual, _replay_fold_own(model, "mode"))
    _assert_differs(actual, _replay_last_fold(model))


def test_fold_unknown_value_under_nan_matches_the_fold_replay() -> None:
    """Passes before H-0114 too: on this data the last pipeline encodes "b" as
    a category the fold model never saw, which LightGBM maps to missing, the
    same input ``nan`` gives the fold. H-0114 asks only for the fold replay here.
    """
    model = _fit(_config("regression", "nan"), _drift_df())
    _assert_affected_folds_split_on_cat(model, "nan")

    _assert_matches(model.importance(kind="shap"), _replay_fold_own(model, "nan"))


# ---------------------------------------------------------------------------
# Acceptance 2: a value the fold knows but the last fold does not (sliding)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("policy", ["mode", "nan"])
def test_value_unknown_to_the_last_fold_is_explained_as_the_fold_saw_it(
    policy: str,
) -> None:
    model = _fit(_config("regression", policy, sliding=True), _sliding_df())
    _assert_affected_folds_split_on_cat(model, policy)

    actual = model.importance(kind="shap")

    _assert_matches(actual, _replay_fold_own(model, policy))
    _assert_differs(actual, _replay_last_fold(model))


# ---------------------------------------------------------------------------
# Acceptance 3: negative controls -- every fold pipeline knows every value
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "features",
    [
        {"unseen_policy": "mode"},
        {"unseen_policy": "mode", "auto_categorical": False, "categorical": ["cat"]},
    ],
    ids=["auto_categorical", "explicit_categorical"],
)
def test_category_typed_column_gives_the_last_fold_value(
    features: dict[str, Any],
) -> None:
    model = _fit(_config("regression", "mode", features=features), _drift_df())
    pipelines = _fold_own_pipelines(model, "mode")
    known = [_fold_categories(p) for p in pipelines]
    assert known[0] and all(k == known[0] for k in known), known
    models = model._get_fit_state().fit_result.models
    assert all(m.importance(kind="split").get("cat", 0) > 0 for m in models)

    _assert_matches(model.importance(kind="shap"), _replay_last_fold(model))


def test_numeric_only_data_gives_the_last_fold_value() -> None:
    df = _drift_df().drop(columns=["cat"])
    model = _fit(_config("regression", "mode"), df)

    _assert_matches(model.importance(kind="shap"), _replay_last_fold(model))


# ---------------------------------------------------------------------------
# Acceptance 4: the recorded per-fold states
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("task", _TASKS)
def test_fit_records_each_fold_pipeline_state(task: str) -> None:
    model = _fit(_config(task, "mode"), _drift_df(task))
    fit_result = model._get_fit_state().fit_result

    states = fit_result.pipeline_state_per_fold

    assert states is not None
    assert len(states) == len(fit_result.splits.outer)
    assert states[-1] == fit_result.pipeline_state
    replayed = [p.get_state() for p in _fold_own_pipelines(model, "mode")]
    assert states == replayed
    assert _fold_categories_of(states[0]) != _fold_categories_of(states[-1])
    for state in states:
        assert state["categorical_cols"] == fit_result.categorical_features


def _fold_categories_of(state: dict[str, Any]) -> list[str]:
    return list(state["encoder"]["categories"].get("cat", []))


# ---------------------------------------------------------------------------
# Acceptance 8: compute_shap_importance stays compatible for direct callers
# ---------------------------------------------------------------------------


def _direct_call_inputs(model: Model) -> tuple[Any, ...]:
    state = model._get_fit_state()
    fr = state.fit_result
    return (
        fr.models,
        state.X,
        fr.splits.outer,
        state.cfg.task,
        fr.feature_names,
        fr.pipeline_state,
        state.provider.build_pipeline_factory(),
    )


def test_seven_positional_arguments_keep_the_single_state_behaviour() -> None:
    model = _fit(_config("regression", "mode"), _drift_df())

    legacy = compute_shap_importance(*_direct_call_inputs(model))

    _assert_matches(legacy, _replay_last_fold(model))


def test_per_fold_states_of_the_wrong_length_are_refused() -> None:
    model = _fit(_config("regression", "mode"), _drift_df())
    states = model._get_fit_state().fit_result.pipeline_state_per_fold
    assert states is not None

    with pytest.raises(ValueError, match="pipeline_state_per_fold"):
        compute_shap_importance(
            *_direct_call_inputs(model), pipeline_state_per_fold=states[:-1]
        )


def test_per_fold_states_without_fold_models_are_refused() -> None:
    """The length check runs before the zero-fold shortcut (PR #327 review r1)."""
    with pytest.raises(ValueError, match="pipeline_state_per_fold"):
        compute_shap_importance(
            [],
            pd.DataFrame({"x": []}),
            [],
            "regression",
            ["x"],
            {},
            pipeline_state_per_fold=[{}],
        )
