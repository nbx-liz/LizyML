"""H-0121 (#280): every model-parameter layer is checked against smart resolution.

Smart resolution runs after the parameter layers are merged and its result
wins, so a native name an active smart parameter writes cannot be set by hand
through any layer: the value is replaced without a word. H-0094 refused that on
``fit(params=)``; ``model.params`` had only a literal-string check for three of
the eighteen spellings, and the provider's default fixed parameters and a
restored ``best_model_params`` had none. Measured at ``6273d39`` (binary,
``lgb.train`` spied): 3 of 18 spellings refused, 12 reached ``lgb.train``
beside the smart value, 3 were overwritten outright (``scale_pos_weight=10``
trained at 0.9512).

H-0121 decision 1 moves the check to ``Model._merge_params``, which ``fit`` and
``tune`` both pass through, and applies it to all four layers against the
effective smart parameters -- the config's values with a tuning result's
``best_smart_params`` laid over them.

The population is derived, not listed: the canonical names come from
``SMART_PARAM_TARGETS`` and every spelling from LightGBM's own alias registry
(``accepted_spellings``). Only the switch that turns each smart parameter on or
off, and a value to write, are declared here, and a test pins that the declared
switches cover exactly the smart parameters that write native names.
"""

from __future__ import annotations

from typing import Any

import pytest

from lizyml import Model
from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.core.types.tuning_result import TuningResult
from lizyml.estimators.lgbm.param_names import accepted_spellings
from lizyml.estimators.lgbm.provider import LGBMProvider
from lizyml.estimators.lgbm.smart_params import SMART_PARAM_TARGETS
from tests._helpers import (
    make_binary_df,
    make_config,
    make_multiclass_df,
    make_regression_df,
)
from tests._train_spy import record_lightgbm_calls

TASKS = ("binary", "multiclass", "regression")
ENTRIES = ("fit", "tune")

#: The model config that switches each smart parameter **on**, per task. A task
#: missing from a smart parameter's entry is one where it writes no native name
#: (``balanced`` gives multiclass a sample weight and is unsupported for
#: regression), so there is no active direction to test there.
ENABLE: dict[str, dict[str, dict[str, Any]]] = {
    "auto_num_leaves": dict.fromkeys(TASKS, {"auto_num_leaves": True}),
    "min_data_in_leaf_ratio": dict.fromkeys(TASKS, {"min_data_in_leaf_ratio": 0.01}),
    "min_data_in_bin_ratio": dict.fromkeys(TASKS, {"min_data_in_bin_ratio": 0.01}),
    "feature_weights": dict.fromkeys(TASKS, {"feature_weights": {"feat_a": 2.0}}),
    "balanced": {"binary": {"balanced": True}},
}

#: The model config under which the same smart parameter writes **nothing**,
#: per task. For ``balanced`` the inactive state differs by task: binary needs
#: it switched off, multiclass writes a sample weight even when it is on, and
#: regression leaves it off by default.
DISABLE: dict[str, dict[str, dict[str, Any]]] = {
    "auto_num_leaves": dict.fromkeys(TASKS, {"auto_num_leaves": False}),
    "min_data_in_leaf_ratio": dict.fromkeys(TASKS, {"min_data_in_leaf_ratio": None}),
    "min_data_in_bin_ratio": dict.fromkeys(TASKS, {"min_data_in_bin_ratio": None}),
    "feature_weights": dict.fromkeys(TASKS, {"feature_weights": None}),
    "balanced": {
        "binary": {"balanced": False},
        "multiclass": {"balanced": True},
        "regression": {},
    },
}

#: A value to write for each canonical name, distinct from what smart
#: resolution would compute, so arriving at ``lgb.train`` is observable.
VALUES: dict[str, Any] = {
    "num_leaves": 12,
    "min_data_in_leaf": 3,
    "min_data_in_bin": 3,
    "feature_contri": [2.0, 1.0],
    "feature_pre_filter": False,
    "scale_pos_weight": 10.0,
}

#: ``(smart parameter, canonical name, spelling)`` for every spelling LightGBM
#: accepts of every native name a smart parameter writes.
SURFACE: list[tuple[str, str, str]] = sorted(
    (smart, canonical, spelling)
    for smart, canonicals in SMART_PARAM_TARGETS.items()
    for canonical in canonicals
    for spelling in accepted_spellings(canonical)
)


def test_the_surface_is_the_one_h0121_counts() -> None:
    """Five smart parameters, six canonical names, eighteen spellings.

    A floor rather than a copy of the population: if LightGBM's alias table
    lost entries, the parametrised cells below would shrink without failing.
    """
    assert set(ENABLE) == set(DISABLE) == set(SMART_PARAM_TARGETS)
    assert set(VALUES) == {c for _, c, _ in SURFACE}
    assert len(SMART_PARAM_TARGETS) == 5
    assert len({c for _, c, _ in SURFACE}) == 6
    assert len(SURFACE) >= 18, SURFACE
    assert any(spelling != canonical for _, canonical, spelling in SURFACE)


ACTIVE_CELLS = [
    pytest.param(
        entry, task, smart, canonical, spelling, id=f"{entry}-{task}-{spelling}"
    )
    for entry in ENTRIES
    for task in TASKS
    for smart, canonical, spelling in SURFACE
    if task in ENABLE[smart]
]

INACTIVE_CELLS = [
    pytest.param(
        entry, task, smart, canonical, spelling, id=f"{entry}-{task}-{spelling}"
    )
    for entry in ENTRIES
    for task in TASKS
    for smart, canonical, spelling in SURFACE
]


def _df(task: str) -> Any:
    if task == "binary":
        return make_binary_df(n=160)
    if task == "multiclass":
        return make_multiclass_df(n=180)
    return make_regression_df(n=160)


def _config(
    task: str, entry: str, smart: dict[str, Any], params: dict[str, Any]
) -> dict[str, Any]:
    cfg = make_config(
        task,
        n_estimators=3,
        n_splits=2,
        tuning_n_trials=1 if entry == "tune" else None,
    )
    cfg["model"].update(smart)
    cfg["model"]["params"].update(params)
    if entry == "tune":
        # One harmless dimension, and no default space: the default space
        # carries ratio dimensions whose per-fold resolution writes
        # `min_data_in_*`, which would overwrite the value under test in the
        # inactive direction for reasons that are not this layer's.
        cfg["tuning"]["optuna"]["space_mode"] = "replace"
        cfg["tuning"]["optuna"]["space"] = {
            "learning_rate": {
                "type": "float",
                "low": 0.05,
                "high": 0.1,
                "category": "model",
            }
        }
    return cfg


def _run(model: Model, entry: str) -> None:
    if entry == "tune":
        model.tune()
    else:
        model.fit()


@pytest.mark.parametrize(
    ("entry", "task", "smart", "canonical", "spelling"), ACTIVE_CELLS
)
def test_a_managed_spelling_in_model_params_is_refused_before_training(
    entry: str, task: str, smart: str, canonical: str, spelling: str
) -> None:
    """Accepting a value smart resolution then replaces is the defect."""
    cfg = _config(task, entry, ENABLE[smart][task], {spelling: VALUES[canonical]})
    model = Model(cfg, data=_df(task))

    with record_lightgbm_calls() as seen, pytest.raises(LizyMLError) as exc:
        _run(model, entry)

    assert exc.value.code is ErrorCode.CONFIG_INVALID
    message = exc.value.user_message
    for expected in (spelling, smart, "model.params"):
        assert expected in message, f"{expected!r} not named in: {message}"
    if spelling != canonical:
        assert canonical in message, message
    assert not seen["train_params"], (
        f"{len(seen['train_params'])} Booster(s) trained before the refusal"
    )


@pytest.mark.parametrize(
    ("entry", "task", "smart", "canonical", "spelling"), INACTIVE_CELLS
)
def test_the_same_spelling_reaches_training_when_its_smart_parameter_is_off(
    entry: str, task: str, smart: str, canonical: str, spelling: str
) -> None:
    """The other direction: without it, refusing everything would pass."""
    value = VALUES[canonical]
    cfg = _config(task, entry, DISABLE[smart][task], {spelling: value})
    model = Model(cfg, data=_df(task))

    with record_lightgbm_calls() as seen:
        _run(model, entry)

    calls = seen["train_params"]
    assert calls, "no lgb.train call was recorded"
    got = [call.get(spelling) for call in calls]
    assert all(v == value for v in got), (
        f"with {smart} inactive, {spelling}={value!r} should reach lgb.train; "
        f"arrived as {got}"
    )


def test_the_config_is_still_constructed() -> None:
    """The refusal moved from config construction to fit/tune (H-0121)."""
    cfg = make_config("binary", n_estimators=3, n_splits=2, num_leaves=12)
    Model(cfg, data=make_binary_df(n=160))


# ---------------------------------------------------------------------------
# The other layers (criterion 2)
# ---------------------------------------------------------------------------


def _tuning_result(
    best_model: dict[str, Any], best_smart: dict[str, Any] | None = None
) -> TuningResult:
    return TuningResult(
        best_model_params=best_model,
        best_smart_params=best_smart or {},
        best_training_params={},
        best_score=0.0,
        metric_name="logloss",
        direction="minimize",
        trials=[],
        rounds=(),
    )


def _with_fixed_collision(monkeypatch: pytest.MonkeyPatch) -> None:
    real = LGBMProvider.default_fixed_params

    def colliding(self: LGBMProvider, task: Any) -> dict[str, Any]:
        return {**real(self, task), "max_leaves": 12}

    monkeypatch.setattr(LGBMProvider, "default_fixed_params", colliding)


def test_default_fixed_params_are_checked_by_tune(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``tune`` itself refuses, naming the provider's fixed layer.

    A fresh study does not overlay the fixed layer in ``_merge_params`` -- the
    objective does, per trial -- so a check that only looked at what
    ``_merge_params`` overlays would let every trial train with both spellings.
    """
    _with_fixed_collision(monkeypatch)
    cfg = make_config("binary", n_estimators=3, n_splits=2, tuning_n_trials=1)
    model = Model(cfg, data=make_binary_df(n=160))

    with record_lightgbm_calls() as seen, pytest.raises(LizyMLError) as exc:
        model.tune()

    assert exc.value.code is ErrorCode.CONFIG_INVALID
    assert "provider default fixed params" in exc.value.user_message
    assert "max_leaves" in exc.value.user_message
    assert not seen["train_params"]

    # The following fit, with a tuning result whose fixed policy is the
    # provider default, meets the same layer and refuses it too.
    model._tuning_result = _tuning_result({})
    with record_lightgbm_calls() as seen, pytest.raises(LizyMLError) as exc:
        model.fit()
    assert exc.value.code is ErrorCode.CONFIG_INVALID
    assert "provider default fixed params" in exc.value.user_message
    assert not seen["train_params"]


def test_a_restored_best_model_params_collision_is_refused() -> None:
    cfg = make_config("binary", n_estimators=3, n_splits=2)
    model = Model(cfg, data=make_binary_df(n=160))
    model._tuning_result = _tuning_result({"max_leaves": 12})

    with record_lightgbm_calls() as seen, pytest.raises(LizyMLError) as exc:
        model.fit()

    assert exc.value.code is ErrorCode.CONFIG_INVALID
    assert "tuning best_model_params" in exc.value.user_message
    assert "max_leaves" in exc.value.user_message
    assert not seen["train_params"]


def test_best_smart_params_activation_is_what_the_check_sees() -> None:
    """The config switches the owner off; the tuning result switches it on.

    An implementation that looked only at the config's smart values would pass
    ``max_leaves`` here, and ``auto_num_leaves`` -- active through the tuning
    result -- would replace it.
    """
    cfg = make_config("binary", n_estimators=3, n_splits=2)
    cfg["model"]["auto_num_leaves"] = False
    model = Model(cfg, data=make_binary_df(n=160))
    model._tuning_result = _tuning_result(
        {"max_leaves": 12}, best_smart={"auto_num_leaves": True}
    )

    with record_lightgbm_calls() as seen, pytest.raises(LizyMLError) as exc:
        model.fit()

    assert exc.value.code is ErrorCode.CONFIG_INVALID
    assert "tuning best_model_params" in exc.value.user_message
    assert "auto_num_leaves" in exc.value.user_message
    assert not seen["train_params"]


def test_best_model_params_reach_training_when_no_smart_parameter_claims_them() -> None:
    """The control for the two cases above."""
    cfg = make_config("binary", n_estimators=3, n_splits=2)
    cfg["model"]["auto_num_leaves"] = False
    model = Model(cfg, data=make_binary_df(n=160))
    model._tuning_result = _tuning_result(
        {"max_leaves": 12}, best_smart={"auto_num_leaves": False}
    )

    with record_lightgbm_calls() as seen:
        model.fit()

    assert seen["train_params"]
    assert all(call.get("max_leaves") == 12 for call in seen["train_params"])
