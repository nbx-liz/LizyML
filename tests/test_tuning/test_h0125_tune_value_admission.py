"""H-0125 policy 3: a tune checks objective and metric values before any study.

A trial used to meet a bad ``objective`` / ``metric`` value only inside the
adapter, where ``CONFIG_INVALID`` became a failed trial (``tuner.py`` catches
it): a fixed layer's bad value failed every trial and ended in
``TUNING_FAILED``; a search space with one bad choice dropped that choice's
trials in silence. The values are now checked with fit's rules in the planning
stage, before a study is created, attached or written (INV-R4):

* criterion 3 -- the fixed layers (``model.params`` and the round's fixed
  parameters), on every LightGBM spelling;
* criterion 4 -- every ``category: model`` dimension of the round's effective
  space whose canonical name is ``objective`` / ``metric``: a numeric one is
  refused (``not_categorical``), a categorical one has every choice checked --
  from the config space, ``tune(space=)``, a resumed round and the provider's
  default space alike.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any
from unittest.mock import patch

import optuna
import pytest

from lizyml import Model
from lizyml.core import _model_tuning
from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.core.types.search_dim import CategoricalDim, SearchDim
from lizyml.estimators.lgbm import provider
from lizyml.estimators.lgbm.defaults import TASK_COMPATIBLE_OBJECTIVES, default_space
from lizyml.estimators.lgbm.provider import LGBMProvider
from tests._helpers import (
    make_binary_df,
    make_config,
    make_multiclass_df,
    make_regression_df,
)
from tests._train_spy import record_lightgbm_calls
from tests.test_tuning._resume_helpers import snapshot_model, snapshot_study, url

CI = ErrorCode.CONFIG_INVALID
LR: dict[str, Any] = {"type": "float", "low": 0.05, "high": 0.3, "log": True}
FIXED_LAYER = "provider default fixed params"


def _config(
    task: str,
    space: dict[str, Any],
    *,
    mode: str = "replace",
    n_trials: int = 2,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    split = "kfold" if task == "regression" else "stratified_kfold"
    cfg = make_config(
        task, n_estimators=3, n_splits=2, split_method=split, tuning_n_trials=n_trials
    )
    cfg["tuning"]["optuna"]["params"].pop("direction")
    cfg["model"]["params"].update(params or {})
    cfg["tuning"]["optuna"]["space_mode"] = mode
    cfg["tuning"]["optuna"]["space"] = copy.deepcopy(space)
    if mode == "merge":
        cfg["tuning"]["optuna"]["space"].setdefault(
            "n_estimators", {"type": "int", "low": 3, "high": 5}
        )
    return cfg


def _frame(task: str) -> Any:
    if task == "binary":
        return make_binary_df(n=120)
    if task == "multiclass":
        return make_multiclass_df(n=150)
    return make_regression_df(n=120)


def _cat(*choices: Any) -> dict[str, Any]:
    return {"type": "categorical", "choices": list(choices)}


def _refused_before_any_study(
    cfg: dict[str, Any], task: str, tmp_path: Path, **tune_kwargs: Any
) -> LizyMLError:
    storage = url(tmp_path)
    with (
        patch("optuna.create_study", wraps=optuna.create_study) as created,
        record_lightgbm_calls() as seen,
        pytest.raises(LizyMLError) as caught,
    ):
        Model(cfg, data=_frame(task)).tune(
            storage=storage, study_name="s", **tune_kwargs
        )
    created.assert_not_called()
    assert seen["train_params"] == []
    assert [
        s.study_name
        for s in optuna.get_all_study_summaries(storage, include_best_trial=False)
    ] == []
    assert caught.value.code == CI
    return caught.value


# ---------------------------------------------------------------------------
# Criterion 3: the fixed layers
# ---------------------------------------------------------------------------

OBJECTIVE_SPELLINGS = ("objective", "objective_type", "app", "application", "loss")
METRIC_SPELLINGS = ("metric", "metrics", "metric_types")

#: ``(task, parameter as written, value, context checks)``.
FIXED_ROWS: list[tuple[str, str, Any, dict[str, Any]]] = [
    *[
        ("binary", key, "cross_entropy_lambda", {"reason": "not_a_probability"})
        for key in OBJECTIVE_SPELLINGS
    ],
    ("binary", "objective", "regression", {"objective": "regression"}),
    ("regression", "objective", "binary", {"objective": "binary"}),
    *[
        (
            "regression",
            key,
            [{"huber": {"delta": 9.0}}],
            {"metric": "huber", "params": ["delta"]},
        )
        for key in METRIC_SPELLINGS
    ],
    ("regression", "metric", ["auc"], {"metric": "auc"}),
    ("binary", "metric", "rmse", {"metric": "rmse"}),
]


@pytest.mark.parametrize(
    ("task", "key", "value", "expected"),
    FIXED_ROWS,
    ids=[f"{r[0]}-{r[1]}-{r[2]!r}" for r in FIXED_ROWS],
)
def test_model_params_value_is_refused_before_any_study(
    task: str,
    key: str,
    value: Any,
    expected: dict[str, Any],
    tmp_path: Path,
) -> None:
    """Replace mode: no fixed layer and no dimension overrides the value, so
    every trial would train on it."""
    cfg = _config(task, {"learning_rate": LR}, params={key: value})
    error = _refused_before_any_study(cfg, task, tmp_path)
    assert error.context["surface"] == "model.params"
    assert error.context["parameter"] == key
    for name, want in expected.items():
        assert error.context[name] == want


@pytest.mark.parametrize(
    ("task", "mode", "params", "space", "canonical", "trained"),
    [
        # A dimension sets the objective in every trial.
        (
            "binary",
            "replace",
            {"objective_type": "cross_entropy_lambda"},
            {"learning_rate": LR, "objective": _cat("binary")},
            "objective",
            "binary",
        ),
        # Merge mode: the fixed layer sets the metric in every trial.
        (
            "regression",
            "merge",
            {"metrics": [{"huber": {"delta": 9.0}}]},
            {"learning_rate": LR},
            "metric",
            ["huber", "mae", "mape"],
        ),
    ],
    ids=["dimension-overrides", "fixed-layer-overrides"],
)
def test_a_value_no_trial_trains_on_is_not_refused(
    task: str,
    mode: str,
    params: dict[str, Any],
    space: dict[str, Any],
    canonical: str,
    trained: Any,
) -> None:
    """The check reads what the trials receive, as fit reads its merged
    parameters (``test_tuning_validates_after_sampled_overlay``): a value a
    later layer or a dimension replaces in every trial reaches no booster."""
    cfg = _config(task, space, mode=mode, params=params)
    with record_lightgbm_calls() as seen:
        Model(cfg, data=_frame(task)).tune()
    assert seen["train_params"]
    for sent in seen["train_params"]:
        assert sent[canonical] == trained
        assert not set(params) & set(sent)


class _BadFixedProvider(LGBMProvider):
    """The shipped provider whose default fixed metric carries native params."""

    def default_fixed_params(self, task: Any) -> dict[str, Any]:
        return {**super().default_fixed_params(task), "metric": [{"huber": {"x": 1}}]}


def test_fixed_layer_value_is_refused_before_any_study(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(_model_tuning, "get_provider", lambda _c: _BadFixedProvider())
    cfg = _config("regression", {"learning_rate": LR}, mode="merge")
    error = _refused_before_any_study(cfg, "regression", tmp_path)
    assert error.context["surface"] == FIXED_LAYER
    assert error.context["parameter"] == "metric"
    assert error.context["params"] == ["x"]


# ---------------------------------------------------------------------------
# Criterion 4: the search space's objective / metric dimensions
# ---------------------------------------------------------------------------

#: ``(task, dimension name, spec, refused value or None, reason or None)``.
DIM_ROWS: list[tuple[str, str, dict[str, Any], Any, str | None]] = [
    (
        "binary",
        "objective",
        _cat("binary", "cross_entropy_lambda"),
        "cross_entropy_lambda",
        "not_a_probability",
    ),
    ("binary", "objective_type", _cat("binary", "regression"), "regression", None),
    (
        "binary",
        "application",
        _cat("cross_entropy", "cross_entropy_lambda"),
        "cross_entropy_lambda",
        "not_a_probability",
    ),
    ("regression", "loss", _cat("huber", "multiclass"), "multiclass", None),
    ("binary", "metric", _cat("auc", "rmse"), "rmse", None),
    ("regression", "metrics", _cat("rmse", "auc"), "auc", None),
    ("multiclass", "metric_types", _cat("multi_logloss", "nope"), "nope", None),
    (
        "binary",
        "objective",
        {"type": "int", "low": 1, "high": 2},
        None,
        "not_categorical",
    ),
    (
        "regression",
        "metric",
        {"type": "float", "low": 0.0, "high": 1.0},
        None,
        "not_categorical",
    ),
    ("binary", "app", {"type": "int", "low": 1, "high": 2}, None, "not_categorical"),
]


def _assert_dim_refusal(
    error: LizyMLError,
    *,
    surface: str,
    name: str,
    value: Any,
    reason: str | None,
) -> None:
    assert error.context["surface"] == surface
    assert error.context["dim"] == name
    assert f"'{name}'" in error.user_message
    assert error.user_message.startswith(f"{surface}: ")
    if value is not None:
        assert error.context["key"] == "choices"
        assert error.context["value"] == value
    if reason is not None:
        assert error.context["reason"] == reason


@pytest.mark.parametrize(
    ("task", "name", "spec", "value", "reason"),
    DIM_ROWS,
    ids=[f"{r[0]}-{r[1]}-{r[2]['type']}" for r in DIM_ROWS],
)
@pytest.mark.parametrize("mode", ["replace", "merge"])
def test_config_space_dimension_is_refused_before_any_study(
    task: str,
    name: str,
    spec: dict[str, Any],
    value: Any,
    reason: str | None,
    mode: str,
    tmp_path: Path,
) -> None:
    cfg = _config(task, {"learning_rate": LR, name: spec}, mode=mode)
    error = _refused_before_any_study(cfg, task, tmp_path)
    _assert_dim_refusal(
        error, surface="tuning.optuna.space", name=name, value=value, reason=reason
    )


@pytest.mark.parametrize(
    ("task", "name", "choices"),
    [
        ("binary", "objective", ["binary", "cross_entropy"]),
        ("regression", "objective", ["huber", "regression_l1"]),
        ("binary", "metric", ["auc", "binary_logloss"]),
        ("regression", "metric", ["rmse", "l1"]),
    ],
)
def test_valid_choices_still_tune(task: str, name: str, choices: list[str]) -> None:
    cfg = _config(task, {"learning_rate": LR, name: _cat(*choices)}, n_trials=3)
    with record_lightgbm_calls() as seen:
        result = Model(cfg, data=_frame(task)).tune()
    assert all(t.state == "complete" for t in result.trials)
    canonical = "objective" if name == "objective" else "metric"
    sent = {
        str(v)
        for p in seen["train_params"]
        for v in (p[canonical] if isinstance(p[canonical], list) else [p[canonical]])
    }
    assert sent <= set(choices)


def test_the_provider_default_space_offers_no_cross_entropy_lambda() -> None:
    provider = LGBMProvider()
    dims = default_space("binary", provider=provider)
    objective = next(d for d in dims if d.name == "objective")
    assert isinstance(objective, CategoricalDim)
    assert objective.choices == ("binary", "cross_entropy")


class _BadDefaultSpaceProvider(LGBMProvider):
    """The shipped provider whose default ``objective`` offers another task's."""

    def default_space(self, task: Any) -> list[SearchDim]:
        out: list[SearchDim] = []
        for dim in super().default_space(task):
            if dim.name == "objective":
                dim = CategoricalDim("objective", ("binary", "huber"), category="model")
            out.append(dim)
        return out


def test_a_default_space_dimension_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        _model_tuning, "get_provider", lambda _c: _BadDefaultSpaceProvider()
    )
    cfg = _config(
        "binary", {"n_estimators": {"type": "int", "low": 3, "high": 5}}, mode="merge"
    )
    error = _refused_before_any_study(cfg, "binary", tmp_path)
    _assert_dim_refusal(
        error,
        surface="tuning.optuna.space",
        name="objective",
        value="huber",
        reason=None,
    )


# ---------------------------------------------------------------------------
# Criterion 4: later rounds
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("spec", "value", "reason"),
    [
        (_cat("binary", "regression"), "regression", None),
        (
            _cat("binary", "cross_entropy_lambda"),
            "cross_entropy_lambda",
            "not_a_probability",
        ),
        ({"type": "int", "low": 1, "high": 2}, None, "not_categorical"),
    ],
    ids=["other-task", "not-a-probability", "numeric"],
)
def test_a_space_argument_dimension_is_refused_before_any_write(
    spec: dict[str, Any], value: Any, reason: str | None, tmp_path: Path
) -> None:
    storage = url(tmp_path)
    cfg = _config("binary", {"learning_rate": LR})
    model = Model(cfg, data=_frame("binary"))
    model.tune(storage=storage, study_name="s")
    before = (snapshot_study(storage, "s"), snapshot_model(model))
    with record_lightgbm_calls() as seen, pytest.raises(LizyMLError) as caught:
        model.tune(
            resume=True,
            storage=storage,
            study_name="s",
            space={"learning_rate": LR, "objective": spec},
        )
    assert caught.value.code == CI
    _assert_dim_refusal(
        caught.value,
        surface="tune(space=)",
        name="objective",
        value=value,
        reason=reason,
    )
    assert seen["train_params"] == []
    assert (snapshot_study(storage, "s"), snapshot_model(model)) == before


def test_a_resumed_round_with_a_stored_bad_choice_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A study written before the rule (cross_entropy_lambda was accepted)."""
    storage = url(tmp_path)
    cfg = _config(
        "binary",
        {"learning_rate": LR, "objective": _cat("binary", "cross_entropy_lambda")},
    )
    with monkeypatch.context() as patched:
        # The two tables of v0.18.0 (provider.py checks they agree).
        patched.setitem(
            TASK_COMPATIBLE_OBJECTIVES,
            "binary",
            frozenset({"binary", "cross_entropy", "cross_entropy_lambda"}),
        )
        patched.setitem(
            provider._LGBM_OBJECTIVE_CHOICES,
            "binary",
            ("binary", "cross_entropy", "cross_entropy_lambda"),
        )
        Model(cfg, data=_frame("binary")).tune(storage=storage, study_name="s")
    before = snapshot_study(storage, "s")
    model = Model(cfg, data=_frame("binary"))
    with record_lightgbm_calls() as seen, pytest.raises(LizyMLError) as caught:
        model.tune(resume=True, storage=storage, study_name="s")
    _assert_dim_refusal(
        caught.value,
        surface="tuning.optuna.space",
        name="objective",
        value="cross_entropy_lambda",
        reason="not_a_probability",
    )
    assert seen["train_params"] == []
    assert snapshot_study(storage, "s") == before
