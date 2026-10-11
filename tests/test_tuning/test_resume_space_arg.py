"""H-0122 §5: ``tune(space=...)`` on resume (PR 2).

Each test runs real tunes and reads the round records the study holds. Every
refusal also checks INV-R4 the criterion-22 way: the study (trials and every
``lizyml:`` attribute) and the Model's tune-owned state are unchanged.

Criteria: 4, 6, 6b, 7, 12 (category), 15 (``space=``), 17, 29, 30 (bullet 1).
"""

from __future__ import annotations

import copy
import logging
import shutil
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest

from lizyml import Model
from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.estimators.lgbm.provider import LGBMProvider
from lizyml.tuning.lineage import classify_study
from lizyml.tuning.search_space import parse_space
from tests._train_spy import record_lightgbm_calls
from tests.test_tuning._resume_helpers import (
    ADMISSION_SPACES,
    CAT_LR,
    SMALL_TREES,
    ProcessDeath,
    cat_config,
    die_in_objective,
    fast_config,
    frame,
    round_record,
    round_trials,
    snapshot_model,
    snapshot_study,
    space_of,
    spy_fit_params,
    url,
)

CI = ErrorCode.CONFIG_INVALID
LR = {"type": "float", "low": 0.05, "high": 0.3, "log": True}
#: A range far below anything useful with 3 trees: no round-2 trial can beat
#: round 1 (design R-11; each test still asserts that precondition).
BAD_LR = {"type": "float", "low": 1e-4, "high": 2e-4, "log": True}


def _refused(model: Model, study: Any, **kwargs: Any) -> LizyMLError:
    """Call ``tune``; assert CONFIG_INVALID and that nothing changed (INV-R4)."""
    before = (snapshot_study(study), snapshot_model(model))
    with pytest.raises(LizyMLError) as caught:
        model.tune(**kwargs)
    assert caught.value.code is CI
    assert (snapshot_study(study), snapshot_model(model)) == before
    return caught.value


def _best_from_round_1(model: Model) -> None:
    """Precondition (R-11): the study's best trial is a round-1 trial."""
    study = model._study
    assert study is not None
    start_2 = round_record(study, 2).start_trial
    assert study.best_trial.number < start_2


def _dim(record: Any, name: str, which: str = "effective") -> Any:
    return next(d for d in getattr(record, which) if d.name == name)


# ---------------------------------------------------------------------------
# Criterion 4: a moved range samples the new range and keeps the old best
# ---------------------------------------------------------------------------


def test_moved_range_samples_new_range_and_keeps_old_best(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = fast_config(n_trials=3)
    model = Model(cfg, data=frame())
    first = model.tune()
    space = space_of(cfg)
    space["learning_rate"] = BAD_LR
    result = model.tune(resume=True, space=space, expand_boundary=True)

    _best_from_round_1(model)
    for trial in round_trials(model._study, 2):
        assert 1e-4 <= trial.params["learning_rate"] <= 2e-4
    assert result.best_params["learning_rate"] == first.best_params["learning_rate"]
    assert result.best_score == first.best_score
    record = round_record(model._study, 2)
    assert record.source == "space_arg"
    assert record.space_arg is not None
    assert [d.name for d in record.space_arg] == list(space)
    lr = _dim(record, "learning_rate")
    assert (lr.low, lr.high) == (1e-4, 2e-4)
    row = model.boundary_table().set_index("dim").loc["learning_rate"]
    assert bool(row["expanded"]) is False

    seen = spy_fit_params(monkeypatch, model)
    model.fit()
    assert (
        seen[-1]["model_params"]["learning_rate"] == first.best_params["learning_rate"]
    )


# ---------------------------------------------------------------------------
# Criteria 6 and 6b: a removed dim keeps its route
# ---------------------------------------------------------------------------


def _ratio_config() -> dict[str, Any]:
    cfg = fast_config(n_trials=3)
    cfg["model"]["auto_num_leaves"] = True
    cfg["tuning"]["optuna"]["space"] = {
        "learning_rate": dict(LR),
        "num_leaves_ratio": {
            "type": "float",
            "low": 0.5,
            "high": 0.9,
            "category": "smart",
        },
    }
    return cfg


def test_removed_smart_dim_routes_to_smart(monkeypatch: pytest.MonkeyPatch) -> None:
    """Criterion 6: later trials use the config value; the best's goes to smart."""
    model = Model(_ratio_config(), data=frame())
    model.tune()
    config_ratio = model._cfg.model.num_leaves_ratio
    trial_calls = spy_fit_params(monkeypatch, model)
    result = model.tune(resume=True, space={"learning_rate": BAD_LR})
    monkeypatch.undo()

    _best_from_round_1(model)
    assert trial_calls
    for call in trial_calls:
        assert call["smart_params"]["num_leaves_ratio"] == config_ratio
    best_ratio = model._study.best_trial.params["num_leaves_ratio"]
    assert result.best_smart_params["num_leaves_ratio"] == best_ratio
    assert "num_leaves_ratio" not in result.best_model_params

    seen = spy_fit_params(monkeypatch, model)
    with record_lightgbm_calls() as calls:
        model.fit()
    assert seen[-1]["smart_params"]["num_leaves_ratio"] == best_ratio
    assert calls["train_params"]
    for params in calls["train_params"]:
        assert "num_leaves_ratio" not in params


def test_removed_training_dim_stays_training(monkeypatch: pytest.MonkeyPatch) -> None:
    """Criterion 6b: ``early_stopping_rounds`` removed, the best keeps it."""
    cfg = fast_config(n_trials=3)
    cfg["tuning"]["optuna"]["space"] = {
        "learning_rate": dict(LR),
        "early_stopping_rounds": {
            "type": "int",
            "low": 5,
            "high": 10,
            "category": "training",
        },
    }
    model = Model(cfg, data=frame())
    model.tune()
    result = model.tune(resume=True, space={"learning_rate": BAD_LR})

    _best_from_round_1(model)
    best_esr = model._study.best_trial.params["early_stopping_rounds"]
    assert result.best_training_params == {"early_stopping_rounds": best_esr}
    assert "early_stopping_rounds" not in result.best_model_params
    for trial in round_trials(model._study, 2):
        assert "early_stopping_rounds" not in trial.params

    seen = spy_fit_params(monkeypatch, model)
    model.fit()
    assert seen[-1]["training_overrides"] == {"early_stopping_rounds": best_esr}


# ---------------------------------------------------------------------------
# Criteria 7 and 12: changes one study cannot take
# ---------------------------------------------------------------------------


#: Shared with the criterion-22 table (`_resume_helpers.cat_config`).
_cat_config = cat_config
assert CAT_LR == LR


UNCHANGEABLE = {
    "kind": ("learning_rate", {"type": "int", "low": 1, "high": 3}, "kind"),
    "log": ("learning_rate", {**LR, "log": False}, "log"),
    "choices_true_to_1": (
        "extra_trees",
        {"type": "categorical", "choices": [1, 0]},
        "choices",
    ),
    "choices_1_to_1p0": (
        "lambda_l2",
        {"type": "categorical", "choices": [0.0, 1.0]},
        "choices",
    ),
    "choices_order": (
        "lambda_l2",
        {"type": "categorical", "choices": [1, 0]},
        "choices",
    ),
}


@pytest.mark.parametrize("case", list(UNCHANGEABLE))
def test_unchangeable_refused(case: str) -> None:
    """Criterion 7: refused before the study changes (Optuna would FAIL a
    trial for kind / log / order, and pass the type-only changes)."""
    name, spec, changed = UNCHANGEABLE[case]
    cfg = _cat_config()
    model = Model(cfg, data=frame())
    model.tune()
    space = space_of(cfg)
    space[name] = spec
    err = _refused(model, model._study, resume=True, space=space)
    assert err.context["surface"] == "tune(space=)"
    assert err.context["dim"] == name
    assert err.context["changed"] == changed


def test_range_only_change_passes() -> None:
    """The control for criterion 7: low / high may change."""
    cfg = _cat_config()
    model = Model(cfg, data=frame())
    model.tune()
    space = space_of(cfg)
    space["learning_rate"] = {**LR, "low": 0.01}
    model.tune(resume=True, space=space)
    lr = _dim(round_record(model._study, 2), "learning_rate", "declared")
    assert lr.low == 0.01


def test_space_mode_edit_with_space_arg_is_the_config_refusal(
    tmp_path: Path,
) -> None:
    """Criterion 7 (``space_mode``): ``space=`` does not let the mode change."""
    storage = url(tmp_path)
    cfg = fast_config()
    Model(cfg, data=frame()).tune(storage=storage, study_name="s")
    edited = copy.deepcopy(cfg)
    edited["tuning"]["optuna"]["space_mode"] = "merge"
    model = Model(edited, data=frame())
    before = (snapshot_study(storage, "s"), snapshot_model(model))
    with pytest.raises(LizyMLError) as caught:
        model.tune(resume=True, storage=storage, study_name="s", space=space_of(cfg))
    assert caught.value.code is CI
    assert caught.value.context["surface"] == "tuning.optuna.space_mode"
    assert (snapshot_study(storage, "s"), snapshot_model(model)) == before


@pytest.mark.parametrize(
    "name", ["num_leaves_ratio", "early_stopping_rounds"], ids=["smart", "training"]
)
def test_category_change_refused(name: str) -> None:
    """Criterion 12: a routed name re-declared as ``model`` (4.1b message)."""
    cfg = _cat_config()
    model = Model(cfg, data=frame())
    model.tune()
    space = space_of(cfg)
    space[name] = {**space[name], "category": "model"}
    err = _refused(model, model._study, resume=True, space=space)
    assert err.context["surface"] == "tune(space=)"
    assert "category" in err.user_message
    assert name in err.context["reason"]


# ---------------------------------------------------------------------------
# Criterion 15: a space= round has no enqueued trial
# ---------------------------------------------------------------------------


def test_space_round_has_no_enqueue() -> None:
    cfg = fast_config(n_trials=3)
    model = Model(cfg, data=frame())
    model.tune()
    space = space_of(cfg)
    space["learning_rate"] = {**LR, "high": 0.5}
    result = model.tune(resume=True, space=space)
    study = model._study
    round_2 = round_trials(study, 2)
    assert len(round_2) == 3
    for trial in round_2:
        assert "fixed_params" not in trial.system_attrs
    assert result.best_score == study.best_value
    assert result.best_params == study.best_trial.params


# ---------------------------------------------------------------------------
# Criterion 17: expand_boundary=None after a space= user dim, on reattach
# ---------------------------------------------------------------------------


def test_expand_none_after_user_dim_same_after_reattach(tmp_path: Path) -> None:
    """Round 1 uses the defaults only (H-0102: expand), round 2 adds user
    dims with ``space=``, round 3 asks ``expand_boundary=None``.

    The decision must come from round 2's recorded provenance: a reattached
    Model whose config space has no user dim must not re-derive it.
    """
    cfg = fast_config(mode="merge", n_trials=1)
    cfg["tuning"]["optuna"]["space"] = {}
    storage = url(tmp_path / "a")
    (tmp_path / "a").mkdir()
    model = Model(cfg, data=frame())
    model.tune(storage=storage, study_name="s")
    assert round_record(storage, 1).expand_boundary is True
    space = {
        "n_estimators": dict(SMALL_TREES),
        "max_depth": {"type": "int", "low": 3, "high": 6},
    }
    model.tune(resume=True, space=space, expand_boundary=False)

    (tmp_path / "b").mkdir()
    shutil.copy(tmp_path / "a" / "s.db", tmp_path / "b" / "s.db")
    copy_url = url(tmp_path / "b")

    model.tune(resume=True)  # variant A: the same Model
    Model(cfg, data=frame()).tune(resume=True, storage=copy_url, study_name="s")

    in_process, reattached = round_record(storage, 3), round_record(copy_url, 3)
    assert in_process.expand_boundary is False
    assert reattached.expand_boundary is False
    assert in_process.boundary_report == reattached.boundary_report
    assert [d.name for d in in_process.effective] == [
        d.name for d in reattached.effective
    ]


# ---------------------------------------------------------------------------
# Criterion 29: space= and the admission checks
# ---------------------------------------------------------------------------


def test_merge_space_changes_one_default_keeps_others() -> None:
    """29.1: the other default dims stay declared and are sampled."""
    cfg = fast_config(mode="merge", n_trials=2)
    model = Model(cfg, data=frame())
    model.tune()
    space = {"n_estimators": dict(SMALL_TREES), "learning_rate": {**LR, "high": 0.2}}
    model.tune(resume=True, space=space)
    record = round_record(model._study, 2)
    provider_names = {d.name for d in LGBMProvider().default_space("regression")} - {
        "num_leaves_ratio"
    }  # dropped by H-0121 (auto_num_leaves is off)
    assert {d.name for d in record.declared} == provider_names
    origins = {p.name: p.origin for p in record.provenance}
    assert origins.pop("learning_rate") == "user"
    assert origins.pop("n_estimators") == "user"
    assert set(origins.values()) == {"default"}
    for trial in round_trials(model._study, 2):
        assert set(trial.params) == provider_names


def test_merge_space_omitting_config_user_dim_reverts_to_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """29.1, the sharp side (design R-1): merge ``space=`` declares the
    provider defaults plus ``space=``; a config user dim it omits reverts to
    its default range, is a changed declaration and is not expanded."""
    cfg = fast_config(mode="merge", n_trials=2)
    model = Model(cfg, data=frame())
    model.tune()
    study = model._study
    die_in_objective(monkeypatch, model, at_call=1)
    with pytest.raises(ProcessDeath):
        model.tune(resume=True, space={"learning_rate": LR}, expand_boundary=True)
    record = round_record(study, 2)
    n_est = _dim(record, "n_estimators", "declared")
    assert (n_est.low, n_est.high) == (600, 2500)
    assert {p.name: p.origin for p in record.provenance}["n_estimators"] == "default"
    assert record.boundary_report is not None
    status = {s.name: s for s in record.boundary_report.dims}["n_estimators"]
    assert status.expanded is False


#: Shared with the criterion-22 table (`_resume_helpers.ADMISSION_SPACES`).
ADMISSION = ADMISSION_SPACES


@pytest.mark.parametrize("case", list(ADMISSION))
def test_admission_refused(case: str) -> None:
    """29.2: the new-tune admission checks read the ``space=`` declaration."""
    cfg = fast_config()
    model = Model(cfg, data=frame())
    model.tune()
    err = _refused(model, model._study, resume=True, space=ADMISSION[case])
    (name,) = [n for n in ADMISSION[case] if n != "learning_rate"]
    assert name in err.user_message
    # Every row names the surface the dimension was declared on, H-0121's
    # inactive-smart refusal included (not `tuning.optuna.space`).
    assert "tune(space=)" in err.user_message
    assert "tuning.optuna.space:" not in err.user_message
    if case in {"inactive_smart_user", "unknown_training", "unknown_smart"}:
        # The rows whose context carries a surface.
        assert err.context["surface"] == "tune(space=)"


MALFORMED: dict[str, Any] = {
    "not_a_mapping": [("learning_rate", LR)],
    "dim_not_a_mapping": {"learning_rate": [0.1, 0.2]},
    "name_not_str": {1: LR},
    "unknown_category": {"learning_rate": {**LR, "category": "booster"}},
    "missing_low": {"learning_rate": {"type": "float", "high": 0.2}},
    "unknown_type": {"learning_rate": {"type": "uniform", "low": 0.1, "high": 0.2}},
}


@pytest.mark.parametrize("case", list(MALFORMED))
def test_malformed_space_refused(case: str) -> None:
    model = Model(fast_config(), data=frame())
    model.tune()
    _refused(model, model._study, resume=True, space=MALFORMED[case])


@pytest.mark.parametrize("variant", ["in_process", "reattach"])
def test_h0121_drop_on_space_rounds(
    variant: str, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """29.3: a merge ``space=`` re-adds ``num_leaves_ratio``; H-0121 drops it."""
    storage = url(tmp_path)
    cfg = fast_config(mode="merge", n_trials=1)
    model = Model(cfg, data=frame())
    model.tune(storage=storage, study_name="s")
    assert "num_leaves_ratio" not in {d.name for d in round_record(storage, 1).declared}
    if variant == "reattach":
        model = Model(cfg, data=frame())
    space = {"n_estimators": dict(SMALL_TREES)}
    with caplog.at_level(logging.INFO):
        model.tune(resume=True, storage=storage, study_name="s", space=space)
    assert "tune.drop_dims" in caplog.text
    declared = {d.name for d in round_record(storage, 2).declared}
    assert "num_leaves_ratio" not in declared
    assert "min_data_in_leaf_ratio" in declared


def test_user_ratio_with_a_true_owner_is_kept() -> None:
    """29.3: a ``space=`` ratio whose owner dim can be true is not dropped."""
    cfg = fast_config(mode="merge", n_trials=1)
    model = Model(cfg, data=frame())
    model.tune()
    space = {
        "n_estimators": dict(SMALL_TREES),
        "auto_num_leaves": {
            "type": "categorical",
            "choices": [True, False],
            "category": "smart",
        },
        "num_leaves_ratio": {
            "type": "float",
            "low": 0.5,
            "high": 1.0,
            "category": "smart",
        },
    }
    model.tune(resume=True, space=space)
    declared = {d.name for d in round_record(model._study, 2).declared}
    assert {"num_leaves_ratio", "auto_num_leaves"} <= declared


def test_out_of_bound_user_range_not_clamped() -> None:
    """29.4: a ``space=`` range below the provider bound is sampled as given;
    expansion afterwards never crosses back past it (#342)."""
    cfg = fast_config(mode="merge", n_trials=2)
    model = Model(cfg, data=frame())
    model.tune()
    space = {"n_estimators": {"type": "int", "low": 3, "high": 5}}
    model.tune(resume=True, space=space)
    for trial in round_trials(model._study, 2):
        assert 3 <= trial.params["n_estimators"] <= 5
    model.tune(resume=True, expand_boundary=True, boundary_threshold=0.49)
    n_est = _dim(round_record(model._study, 3), "n_estimators")
    assert n_est.low <= 3 < 5 <= n_est.high
    assert n_est.high <= 10000
    for trial in round_trials(model._study, 3):
        assert trial.state.name == "COMPLETE"


def test_alias_then_canonical_one_route_inherits() -> None:
    """29.5: ``eta`` then ``learning_rate`` is one dim, one route; the
    expansion carries over and the trials keep the ``eta`` key."""
    cfg = fast_config(n_trials=2)
    eta = {"type": "float", "low": 0.05, "high": 0.3}
    cfg["tuning"]["optuna"]["space"] = {
        "eta": eta,
        "num_leaves": {"type": "int", "low": 4, "high": 8},
    }
    model = Model(cfg, data=frame())
    model.tune()
    model.tune(resume=True, expand_boundary=True, boundary_threshold=0.49)
    expanded = _dim(round_record(model._study, 2), "eta")
    assert (expanded.low, expanded.high) != (0.05, 0.3)
    space = {
        "learning_rate": eta,
        "num_leaves": {"type": "int", "low": 4, "high": 8},
    }
    model.tune(resume=True, space=space, expand_boundary=False)
    study = model._study
    record = round_record(study, 3)
    assert _dim(record, "eta") == expanded
    assert record.routes_added == ()
    lineage = classify_study(study.user_attrs, study.trials)
    assert [
        r.name for r in lineage.routes.values() if r.canonical == "learning_rate"
    ] == ["eta"]
    for trial in round_trials(study, 3):
        assert "eta" in trial.params
        assert "learning_rate" not in trial.params


# ---------------------------------------------------------------------------
# Criterion 30, bullet 1: what a new round does not expand
# ---------------------------------------------------------------------------


def test_changed_dim_near_edge_not_expanded() -> None:
    cfg = fast_config(n_trials=3)
    model = Model(cfg, data=frame())
    first = model.tune()
    best = first.best_params["learning_rate"]
    space = space_of(cfg)
    space["learning_rate"] = {**LR, "low": best * 0.99, "high": best * 3}
    result = model.tune(resume=True, space=space, expand_boundary=True)
    assert result.boundary_report is not None
    status = {s.name: s for s in result.boundary_report.dims}["learning_rate"]
    # Round 2's report is decided at its start, from round 1's best.
    assert status.best_value == best
    assert status.edge == "lower"
    assert status.expanded is False
    row = model.boundary_table().set_index("dim").loc["learning_rate"]
    assert bool(row["expanded"]) is False


def test_best_outside_range_not_expanded_next_round() -> None:
    """Round 2 moves the range away (frozen as changed); round 3 repeats the
    declaration, and the study's best (round 1) lies outside it."""
    cfg = fast_config(n_trials=3)
    model = Model(cfg, data=frame())
    first = model.tune()
    space = space_of(cfg)
    space["learning_rate"] = BAD_LR
    model.tune(resume=True, space=space, expand_boundary=False)
    _best_from_round_1(model)
    result = model.tune(resume=True, expand_boundary=True)
    assert result.best_params["learning_rate"] == first.best_params["learning_rate"]
    assert result.boundary_report is not None
    status = {s.name: s for s in result.boundary_report.dims}["learning_rate"]
    assert status.edge == "upper"  # the best is above the moved range
    assert status.expanded is False
    lr = _dim(round_record(model._study, 3), "learning_rate")
    assert (lr.low, lr.high) == (1e-4, 2e-4)


def test_space_arg_record_keeps_the_written_form() -> None:
    """Design D-a: ``space_arg`` holds the argument as parsed -- raw names
    (an alias included) and order -- while ``declared`` is normalised."""
    cfg = fast_config(n_trials=1)
    model = Model(cfg, data=frame())
    model.tune()
    space = {
        "num_leaves": {"type": "int", "low": 4, "high": 8},
        "learning_rate": LR,
    }
    model.tune(resume=True, space=space)
    record = round_record(model._study, 2)
    assert record.space_arg == tuple(parse_space(space))


def test_a_read_only_mapping_is_a_space_arg() -> None:
    """``space=`` takes any mapping; a non-dict spec keeps its category."""
    cfg = fast_config(n_trials=1)
    cfg["model"]["auto_num_leaves"] = True
    cfg["tuning"]["optuna"]["space"] = {"learning_rate": dict(LR)}
    model = Model(cfg, data=frame())
    model.tune()
    ratio = MappingProxyType(
        {"type": "float", "low": 0.5, "high": 1.0, "category": "smart"}
    )
    model.tune(
        resume=True,
        space=MappingProxyType({"learning_rate": LR, "num_leaves_ratio": ratio}),
    )
    record = round_record(model._study, 2)
    assert {(d.name, d.category) for d in record.declared} == {
        ("learning_rate", "model"),
        ("num_leaves_ratio", "smart"),
    }
