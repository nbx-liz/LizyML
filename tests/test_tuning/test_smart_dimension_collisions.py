"""H-0121 amendment 1 (#341): a smart dimension can activate a collision.

A ``category: smart`` dimension overlays its sampled value on every trial, so a
smart parameter the config leaves inactive can be active inside a trial and
manage a name a parameter layer sets. ``check_round_space`` checks every layer
that reaches trials -- ``model.params`` and the round's fixed parameters --
against every combination of the values the round's smart dimensions can take,
reduced to one representative per state (``None`` / falsy / truthy), and
``validate_tuning_dimensions`` checks the ``category: model`` dimensions against
the same product.

Criteria (HISTORY.md H-0121 amendment 1): 1, 2, 3, 4, 4b, 4c, 5, 5b, 5c, 6.
Criterion 5d (the provider contract) is
``tests/test_estimators/test_smart_managed_contract.py``.
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
from lizyml.core._smart_combinations import (
    FIXED_LAYER,
    check_layers_over_smart_values,
    smart_value_candidates,
    smart_value_combinations,
)
from lizyml.core._tuning_validation import validate_tuning_dimensions
from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.core.types.search_dim import CategoricalDim, FloatDim, IntDim
from lizyml.estimators.lgbm.provider import LGBMProvider
from lizyml.estimators.provider import SmartValueRule
from tests._helpers import make_binary_df, make_config, make_regression_df
from tests._train_spy import record_lightgbm_calls
from tests.test_tuning._resume_helpers import snapshot_model, snapshot_study, url

CI = ErrorCode.CONFIG_INVALID
#: A real LightGBM parameter no shipped smart parameter writes; the test
#: providers below make it managed under their own activation rules.
HESSIAN = "min_sum_hessian_in_leaf"
SMALL_TREES: dict[str, Any] = {"type": "int", "low": 3, "high": 5}
LR: dict[str, Any] = {"type": "float", "low": 0.05, "high": 0.3, "log": True}


def _smart(choices: list[Any]) -> dict[str, Any]:
    return {"type": "categorical", "choices": choices, "category": "smart"}


def _config(
    task: str,
    space: dict[str, Any],
    *,
    mode: str = "replace",
    model: dict[str, Any] | None = None,
    **params: Any,
) -> dict[str, Any]:
    split = "stratified_kfold" if task == "binary" else "kfold"
    cfg = make_config(
        task,
        n_estimators=3,
        n_splits=2,
        split_method=split,
        tuning_n_trials=1,
        **params,
    )
    cfg["model"].update(model or {})
    cfg["tuning"]["optuna"]["space_mode"] = mode
    cfg["tuning"]["optuna"]["space"] = copy.deepcopy(space)
    return cfg


def _frame(task: str) -> Any:
    return make_binary_df(n=120) if task == "binary" else make_regression_df(n=120)


def _refused(cfg: dict[str, Any], task: str) -> LizyMLError:
    """``tune`` refuses before any study is created or any Booster trained.

    The study factory is wrapped, not replaced, so a tune that is not refused
    runs for real and the failure shows what LightGBM was handed.
    """
    with (
        patch("optuna.create_study", wraps=optuna.create_study) as study,
        record_lightgbm_calls() as seen,
    ):
        try:
            Model(cfg, data=_frame(task)).tune()
        except LizyMLError as exc:
            caught = exc
        else:
            pytest.fail(f"tune was not refused; lgb.train got {seen['train_params']}")
    assert caught.code is CI
    study.assert_not_called()
    assert seen["train_params"] == []
    return caught


def _managed(err: LizyMLError) -> dict[str, Any]:
    (entry,) = err.context["managed"]
    return dict(entry)


# ---------------------------------------------------------------------------
# Test providers: activation rules the shipped provider does not have
# ---------------------------------------------------------------------------


class _PairProvider(LGBMProvider):
    """Manages ``HESSIAN`` only while ``auto_num_leaves`` and ``balanced`` both are."""

    def smart_managed_param_names(
        self, smart: dict[str, Any], task: Any
    ) -> dict[str, tuple[str, str]]:
        managed = super().smart_managed_param_names(smart, task)
        if smart.get("auto_num_leaves") and smart.get("balanced"):
            managed = {**managed, HESSIAN: (HESSIAN, "balanced")}
        return managed


class _FalsyRatioProvider(LGBMProvider):
    """Manages ``HESSIAN`` only while ``num_leaves_ratio`` is falsy (not None).

    Its value rule lets the ratio span zero: the shipped rule, ``(0, 1]``
    (H-0121 amendment 2), refuses a range reaching the falsy state before the
    collision check is reached, so only a provider declaring such a range can
    exercise the reducer's zero end to end.
    """

    def smart_value_rules(self) -> dict[str, SmartValueRule]:
        rules = dict(super().smart_value_rules())
        rules["num_leaves_ratio"] = SmartValueRule(kind="number", low=-2.0, high=1.0)
        return rules

    def smart_managed_param_names(
        self, smart: dict[str, Any], task: Any
    ) -> dict[str, tuple[str, str]]:
        managed = super().smart_managed_param_names(smart, task)
        ratio = smart.get("num_leaves_ratio")
        if ratio is not None and not ratio:
            managed = {**managed, HESSIAN: (HESSIAN, "num_leaves_ratio")}
        return managed


class _FixedLeavesProvider(LGBMProvider):
    """The shipped provider whose default fixed parameters include ``max_leaves``."""

    def default_fixed_params(self, task: Any) -> dict[str, Any]:
        return {**super().default_fixed_params(task), "max_leaves": 12}


class _CountingProvider(LGBMProvider):
    """The shipped provider, counting ``smart_managed_param_names`` calls."""

    def __init__(self) -> None:
        self.calls = 0

    def smart_managed_param_names(
        self, smart: dict[str, Any], task: Any
    ) -> dict[str, tuple[str, str]]:
        self.calls += 1
        return super().smart_managed_param_names(smart, task)


def _use(monkeypatch: pytest.MonkeyPatch, provider: LGBMProvider) -> None:
    """Hand *provider* to ``Model.tune`` (it resolves it at the call)."""
    monkeypatch.setattr(_model_tuning, "get_provider", lambda _cfg: provider)


# ---------------------------------------------------------------------------
# Criteria 1 and 2: the two #341 reproductions
# ---------------------------------------------------------------------------


def test_c1_categorical_owner_dimension_activates_model_params_alias() -> None:
    cfg = _config(
        "binary",
        {"auto_num_leaves": _smart([True])},
        model={"auto_num_leaves": False},
        max_leaves=12,
    )
    err = _refused(cfg, "binary")
    for word in ("max_leaves", "auto_num_leaves", "model.params"):
        assert word in err.user_message
    assert "tuning.optuna.space" in err.user_message
    entry = _managed(err)
    assert entry["name"] == "max_leaves"
    assert entry["smart_param"] == "auto_num_leaves"
    assert entry["surface"] == "model.params"
    assert entry["smart_dimensions"] == ["auto_num_leaves"]
    assert err.context["space_surface"] == "tuning.optuna.space"


def test_c2_default_ratio_dimension_activates_model_params_alias() -> None:
    cfg = _config(
        "binary",
        {"n_estimators": SMALL_TREES},
        mode="merge",
        model={"min_data_in_leaf_ratio": None},
        min_child_samples=3,
    )
    err = _refused(cfg, "binary")
    for word in ("min_child_samples", "min_data_in_leaf_ratio", "model.params"):
        assert word in err.user_message
    entry = _managed(err)
    assert entry["smart_param"] == "min_data_in_leaf_ratio"
    assert entry["smart_dimensions"] == ["min_data_in_leaf_ratio"]


# ---------------------------------------------------------------------------
# Criterion 3: the fixed-parameter layer
# ---------------------------------------------------------------------------


def test_fixed_layer_name_is_the_models() -> None:
    """The two spellings of the fixed layer's name stay one (DC3)."""
    from lizyml.core.model import _FIXED_LAYER

    assert FIXED_LAYER == _FIXED_LAYER


def test_c3_fixed_layer_collision_names_the_fixed_layer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use(monkeypatch, _FixedLeavesProvider())
    cfg = _config(
        "binary",
        {"n_estimators": SMALL_TREES, "auto_num_leaves": _smart([True])},
        mode="merge",
        model={"auto_num_leaves": False},
    )
    err = _refused(cfg, "binary")
    entry = _managed(err)
    assert entry["name"] == "max_leaves"
    assert entry["surface"] == FIXED_LAYER
    assert FIXED_LAYER in err.user_message
    assert entry["smart_dimensions"] == ["auto_num_leaves"]


# ---------------------------------------------------------------------------
# Criterion 4: what is not refused
# ---------------------------------------------------------------------------


def test_c4_dimension_that_cannot_activate_is_accepted_and_layer_reaches_training() -> (
    None
):
    cfg = _config(
        "binary",
        {"auto_num_leaves": _smart([False]), "learning_rate": LR},
        model={"auto_num_leaves": False},
        max_leaves=12,
    )
    with record_lightgbm_calls() as seen:
        Model(cfg, data=_frame("binary")).tune()
    assert seen["train_params"]
    assert all(p["max_leaves"] == 12 for p in seen["train_params"])


# ---------------------------------------------------------------------------
# Criterion 4b: one representative per state keeps the product small
# ---------------------------------------------------------------------------

#: ``LGBMConfig``'s defaults.
_BASE_SMART: dict[str, Any] = {
    "auto_num_leaves": True,
    "num_leaves_ratio": 1.0,
    "min_data_in_leaf_ratio": 0.01,
    "min_data_in_bin_ratio": 0.01,
    "feature_weights": None,
    "balanced": None,
}

#: Three smart dimensions of 1,000 truthy values each, inside the smart value
#: rule (H-0121 amendment 2): the ratios are the smart parameters that can take
#: that many values (the bool ones take two).
_WIDE = (
    CategoricalDim(
        "num_leaves_ratio",
        tuple(i / 1_000 for i in range(1, 1001)),
        category="smart",
    ),
    CategoricalDim(
        "min_data_in_bin_ratio",
        tuple(i / 10_000 for i in range(1, 1001)),
        category="smart",
    ),
    CategoricalDim(
        "min_data_in_leaf_ratio",
        tuple(i / 10_000 for i in range(1, 1001)),
        category="smart",
    ),
)


def test_c4b_wide_categorical_dimensions_reduce_to_one_combination() -> None:
    combos = smart_value_combinations(_WIDE, {"auto_num_leaves": False})
    assert len(combos) == 1
    (combo,) = combos
    assert all(combo[dim.name] in dim.choices for dim in _WIDE)


def test_c4b_checks_call_the_provider_once_per_combination() -> None:
    space = [*_WIDE, FloatDim("learning_rate", 0.01, 0.1, category="model")]
    smart = dict(_BASE_SMART)
    provider = _CountingProvider()
    validate_tuning_dimensions(provider, space, smart, "binary")
    # The base values, then the single combination (one-at-a-time took 3,001
    # calls; the unreduced product would take 10**9).
    assert provider.calls <= 2
    provider.calls = 0
    check_layers_over_smart_values(
        provider,
        [("model.params", {"learning_rate": 0.1})],
        space,
        smart,
        "binary",
        space_surface="tuning.optuna.space",
    )
    assert provider.calls == 1


# ---------------------------------------------------------------------------
# Criterion 4c: None and falsy are different states
# ---------------------------------------------------------------------------


def _state(value: Any) -> Any:
    return None if value is None else bool(value)


@pytest.mark.parametrize("order", ["forward", "reversed"])
def test_c4c_reducer_returns_one_value_per_state(order: str) -> None:
    choices = (False, None, 0, True, 1)
    if order == "reversed":
        choices = tuple(reversed(choices))
    reps = smart_value_candidates(CategoricalDim("balanced", choices, category="smart"))
    assert len(reps) == 3
    assert sorted(map(str, map(_state, reps))) == ["False", "None", "True"]


def test_c4c_numeric_range_spanning_zero_adds_zero() -> None:
    reps = smart_value_candidates(
        FloatDim("num_leaves_ratio", -1.0, 1.0, category="smart")
    )
    assert sorted(map(_state, reps)) == [False, True]
    assert 0 in reps
    positive = smart_value_candidates(IntDim("x", 2, 9, category="smart"))
    assert positive == (2,)


@pytest.mark.parametrize("choices", [[False, None], [None, False]])
def test_c4c_none_choice_activates_balanced_in_either_order(choices: list[Any]) -> None:
    cfg = _config(
        "binary",
        {"balanced": _smart(choices)},
        model={"balanced": False},
        scale_pos_weight=2.0,
    )
    err = _refused(cfg, "binary")
    entry = _managed(err)
    assert entry["name"] == "scale_pos_weight"
    assert entry["smart_dimensions"] == ["balanced"]


# ---------------------------------------------------------------------------
# Criteria 5, 5b, 5c: activation by a combination, and by zero
# ---------------------------------------------------------------------------

#: Tuned on binary: a ``balanced`` dimension that can be true is refused on
#: regression (H-0121 amendment 2), before the collision check is reached.
_PAIR_SPACE = {
    "auto_num_leaves": _smart([False, True]),
    "balanced": _smart([False, True]),
}


def test_c5_layer_collision_needing_two_smart_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use(monkeypatch, _PairProvider())
    cfg = _config(
        "binary",
        _PAIR_SPACE,
        model={"auto_num_leaves": False, "balanced": False},
        **{HESSIAN: 1e-3},
    )
    err = _refused(cfg, "binary")
    entry = _managed(err)
    assert entry["name"] == HESSIAN
    assert entry["surface"] == "model.params"
    assert entry["smart_dimensions"] == ["auto_num_leaves", "balanced"]


def test_c5b_model_dimension_collision_needing_two_smart_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use(monkeypatch, _PairProvider())
    space = {
        **_PAIR_SPACE,
        HESSIAN: {"type": "float", "low": 1e-3, "high": 1e-2, "category": "model"},
    }
    cfg = _config("binary", space, model={"auto_num_leaves": False, "balanced": False})
    err = _refused(cfg, "binary")
    entry = _managed(err)
    assert entry["name"] == HESSIAN
    assert entry["surface"] == "tuning.optuna.space"
    assert entry["smart_dimensions"] == ["auto_num_leaves", "balanced"]


def test_c5c_numeric_range_spanning_zero_reaches_the_falsy_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use(monkeypatch, _FalsyRatioProvider())
    space = {
        "num_leaves_ratio": {
            "type": "float",
            "low": -1.0,
            "high": 1.0,
            "category": "smart",
        }
    }
    cfg = _config(
        "regression", space, model={"auto_num_leaves": True}, **{HESSIAN: 1e-3}
    )
    err = _refused(cfg, "regression")
    entry = _managed(err)
    assert entry["name"] == HESSIAN
    assert entry["smart_param"] == "num_leaves_ratio"
    assert err.context["smart_values"] == {"num_leaves_ratio": 0}


# ---------------------------------------------------------------------------
# Criterion 6: a resumed round's space= (H-0122 INV-R4)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("variant", ["in_process", "reattach"])
def test_c6_space_arg_round_refused_without_changing_study_or_model(
    variant: str, tmp_path: Path
) -> None:
    storage = url(tmp_path)
    cfg = _config(
        "regression",
        {"learning_rate": LR},
        model={"auto_num_leaves": False},
        max_leaves=12,
    )
    model = Model(cfg, data=_frame("regression"))
    model.tune(storage=storage, study_name="s")
    if variant == "reattach":
        model = Model(cfg, data=_frame("regression"))
    space = {"learning_rate": LR, "auto_num_leaves": _smart([True])}
    before = (snapshot_study(storage, "s"), snapshot_model(model))
    with record_lightgbm_calls() as seen, pytest.raises(LizyMLError) as caught:
        model.tune(resume=True, storage=storage, study_name="s", space=space)
    assert caught.value.code is CI
    assert seen["train_params"] == []
    assert (snapshot_study(storage, "s"), snapshot_model(model)) == before
    err = caught.value
    assert "tune(space=)" in err.user_message
    assert err.context["space_surface"] == "tune(space=)"
    entry = _managed(err)
    assert entry["name"] == "max_leaves"
    assert entry["smart_dimensions"] == ["auto_num_leaves"]
