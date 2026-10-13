"""H-0121 amendment 2 (#347): the values a ``category: smart`` dimension takes.

A smart dimension's sampled value becomes the trial's smart parameter as it
is, and the smart-parameter resolver reads it by truthiness and ``None``. So
the provider declares, per smart parameter, the values it can hold (exact
types, the ratio ranges of the config), and a tune refuses a dimension outside
them before any study write (policies 1-2). The provider's expansion bound of
``num_leaves_ratio`` is the config's ``1.0`` (policy 3), and ``restore_tuning``
refuses a stored result outside the rule (policy 5).

Criteria 1-4 and 6. Criterion 5 (artifacts) is
``tests/test_persistence/test_smart_params_artifact.py``; criterion 7 (one
declaration) is ``tests/test_estimators/test_smart_value_rules.py``.
"""

from __future__ import annotations

import copy
import dataclasses
import math
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import patch

import optuna
import pytest

from lizyml import Model
from lizyml.core import _model_tuning, _tuning_plan
from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.core.types.search_dim import FloatDim, SearchDim
from lizyml.estimators.lgbm.provider import LGBMProvider
from lizyml.tuning.tuner import Tuner
from tests._helpers import (
    make_binary_df,
    make_config,
    make_multiclass_df,
    make_regression_df,
)
from tests._train_spy import record_lightgbm_calls
from tests.test_tuning._resume_helpers import (
    round_record,
    snapshot_model,
    snapshot_study,
    url,
)

CI = ErrorCode.CONFIG_INVALID
INVALID = "smart_value_invalid"
UNSUPPORTED = "smart_value_unsupported_task"
LR: dict[str, Any] = {"type": "float", "low": 0.05, "high": 0.3, "log": True}
SMALL_TREES: dict[str, Any] = {"type": "int", "low": 3, "high": 5}


def _cat(*choices: Any) -> dict[str, Any]:
    return {"type": "categorical", "choices": list(choices), "category": "smart"}


def _int(low: int, high: int) -> dict[str, Any]:
    return {"type": "int", "low": low, "high": high, "category": "smart"}


def _float(low: float, high: float) -> dict[str, Any]:
    return {"type": "float", "low": low, "high": high, "category": "smart"}


def _config(
    task: str,
    space: dict[str, Any],
    *,
    mode: str = "replace",
    n_trials: int = 1,
    model: dict[str, Any] | None = None,
) -> dict[str, Any]:
    split = "kfold" if task == "regression" else "stratified_kfold"
    cfg = make_config(
        task, n_estimators=3, n_splits=2, split_method=split, tuning_n_trials=n_trials
    )
    cfg["model"].update(model or {})
    cfg["tuning"]["optuna"]["space_mode"] = mode
    cfg["tuning"]["optuna"]["space"] = copy.deepcopy(space)
    return cfg


def _frame(task: str) -> Any:
    if task == "binary":
        return make_binary_df(n=120)
    if task == "multiclass":
        return make_multiclass_df(n=150)
    return make_regression_df(n=120)


def _same(a: Any, b: Any) -> bool:
    """Equal by type and value, NaN equal to NaN."""
    if type(a) is not type(b):
        return False
    if isinstance(a, float) and math.isnan(a):
        return math.isnan(b)
    return bool(a == b)


# ---------------------------------------------------------------------------
# Criterion 1: the refusal matrix
# ---------------------------------------------------------------------------

#: ``(task, name, spec, key, value, reason)``: one row per value the amendment
#: lists, with the ``key`` / ``value`` the refusal must name.
REFUSED: list[tuple[str, str, dict[str, Any], str, Any, str]] = [
    ("binary", "auto_num_leaves", _cat("false"), "choices", "false", INVALID),
    ("binary", "auto_num_leaves", _cat("true", "false"), "choices", "true", INVALID),
    ("binary", "auto_num_leaves", _cat(0, 1), "choices", 0, INVALID),
    ("binary", "auto_num_leaves", _cat(None), "choices", None, INVALID),
    ("binary", "auto_num_leaves", _int(0, 1), "low", 0, INVALID),
    ("binary", "auto_num_leaves", _float(0.0, 1.0), "low", 0.0, INVALID),
    ("binary", "balanced", _cat("false"), "choices", "false", INVALID),
    ("binary", "balanced", _cat(1), "choices", 1, INVALID),
    ("binary", "balanced", _int(0, 1), "low", 0, INVALID),
    ("regression", "balanced", _cat(True, False), "choices", True, UNSUPPORTED),
    ("regression", "balanced", _cat(True), "choices", True, UNSUPPORTED),
    ("binary", "num_leaves_ratio", _float(-1.0, 1.0), "low", -1.0, INVALID),
    ("binary", "num_leaves_ratio", _float(0.0, 1.0), "low", 0.0, INVALID),
    ("binary", "num_leaves_ratio", _float(0.5, 2.0), "high", 2.0, INVALID),
    ("binary", "num_leaves_ratio", _cat(0.5, None), "choices", None, INVALID),
    ("binary", "num_leaves_ratio", _cat(True), "choices", True, INVALID),
    ("binary", "num_leaves_ratio", _cat("0.5"), "choices", "0.5", INVALID),
    ("binary", "num_leaves_ratio", _cat(math.nan), "choices", math.nan, INVALID),
    ("binary", "num_leaves_ratio", _cat(math.inf), "choices", math.inf, INVALID),
    ("binary", "min_data_in_leaf_ratio", _float(0.0, 0.2), "low", 0.0, INVALID),
    ("binary", "min_data_in_leaf_ratio", _float(0.1, 1.0), "high", 1.0, INVALID),
    ("binary", "min_data_in_leaf_ratio", _cat("0.1"), "choices", "0.1", INVALID),
    ("binary", "min_data_in_leaf_ratio", _cat(1.0), "choices", 1.0, INVALID),
    ("binary", "min_data_in_leaf_ratio", _cat(math.nan), "choices", math.nan, INVALID),
    ("binary", "min_data_in_bin_ratio", _float(0.0, 0.2), "low", 0.0, INVALID),
    ("binary", "min_data_in_bin_ratio", _float(0.5, 1.0), "high", 1.0, INVALID),
    ("binary", "min_data_in_bin_ratio", _cat(0), "choices", 0, INVALID),
    ("binary", "min_data_in_bin_ratio", _cat("0.05"), "choices", "0.05", INVALID),
    ("binary", "feature_weights", _cat(None), "choices", None, INVALID),
    ("binary", "feature_weights", _cat(0.5), "choices", 0.5, INVALID),
    ("binary", "feature_weights", _float(0.1, 1.0), "low", 0.1, INVALID),
]


def _row_id(row: tuple[str, str, dict[str, Any], str, Any, str]) -> str:
    task, name, spec, _key, _value, _reason = row
    shown = (
        repr(spec["choices"])
        if spec["type"] == "categorical"
        else f"{spec['type']}({spec['low']},{spec['high']})"
    )
    return f"{task}-{name}-{shown}"


def _assert_refusal(
    err: LizyMLError,
    *,
    surface: str,
    name: str,
    key: str,
    value: Any,
    reason: str,
) -> None:
    assert err.code is CI, err
    assert err.context.get("reason") == reason, err.context
    assert err.context["surface"] == surface
    assert err.context["dim"] == name
    assert err.context["key"] == key
    assert _same(err.context["value"], value), err.context
    assert name in err.user_message


@pytest.mark.parametrize("row", REFUSED, ids=[_row_id(r) for r in REFUSED])
def test_config_space_refuses_before_any_study(
    row: tuple[str, str, dict[str, Any], str, Any, str], tmp_path: Path
) -> None:
    task, name, spec, key, value, reason = row
    storage = url(tmp_path)
    cfg = _config(task, {name: spec})
    with (
        patch("optuna.create_study", wraps=optuna.create_study) as created,
        record_lightgbm_calls() as seen,
        pytest.raises(LizyMLError) as caught,
    ):
        Model(cfg, data=_frame(task)).tune(storage=storage, study_name="s")
    _assert_refusal(
        caught.value,
        surface="tuning.optuna.space",
        name=name,
        key=key,
        value=value,
        reason=reason,
    )
    created.assert_not_called()
    assert seen["train_params"] == []
    assert [
        s.study_name
        for s in optuna.get_all_study_summaries(storage, include_best_trial=False)
    ] == []


@pytest.fixture(scope="module")
def stored_studies(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """One closed round per task, in a SQLite file each case copies."""
    out: dict[str, Path] = {}
    for task in ("binary", "regression"):
        path = tmp_path_factory.mktemp(f"study-{task}") / "s.db"
        Model(_config(task, {"learning_rate": LR}), data=_frame(task)).tune(
            storage=f"sqlite:///{path}", study_name="s"
        )
        out[task] = path
    return out


@pytest.mark.parametrize("row", REFUSED, ids=[_row_id(r) for r in REFUSED])
def test_space_arg_round_refuses_without_writing(
    row: tuple[str, str, dict[str, Any], str, Any, str],
    stored_studies: dict[str, Path],
    tmp_path: Path,
) -> None:
    task, name, spec, key, value, reason = row
    shutil.copy(stored_studies[task], tmp_path / "s.db")
    storage = url(tmp_path)
    model = Model(_config(task, {"learning_rate": LR}), data=_frame(task))
    before = (snapshot_study(storage, "s"), snapshot_model(model))
    with record_lightgbm_calls() as seen, pytest.raises(LizyMLError) as caught:
        model.tune(
            resume=True,
            storage=storage,
            study_name="s",
            space={"learning_rate": LR, name: spec},
        )
    _assert_refusal(
        caught.value,
        surface="tune(space=)",
        name=name,
        key=key,
        value=value,
        reason=reason,
    )
    assert seen["train_params"] == []
    assert (snapshot_study(storage, "s"), snapshot_model(model)) == before


# ---------------------------------------------------------------------------
# Criterion 2: what is accepted reaches training as chosen
# ---------------------------------------------------------------------------


@pytest.fixture
def smart_seen(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Every smart dict the provider resolves, as handed to it."""
    seen: list[dict[str, Any]] = []
    real = LGBMProvider.resolve_smart_params

    def spy(self: LGBMProvider, smart: dict[str, Any], *args: Any, **kw: Any) -> Any:
        seen.append(dict(smart))
        return real(self, smart, *args, **kw)

    monkeypatch.setattr(LGBMProvider, "resolve_smart_params", spy)
    return seen


#: ``(task, name, spec)`` the amendment lists as accepted.
ACCEPTED: list[tuple[str, str, dict[str, Any]]] = [
    ("binary", "auto_num_leaves", _cat(True, False)),
    ("binary", "balanced", _cat(True, False, None)),
    ("multiclass", "balanced", _cat(True, False, None)),
    ("regression", "balanced", _cat(False, None)),
    ("binary", "num_leaves_ratio", _float(0.5, 1.0)),
    ("binary", "num_leaves_ratio", _cat(0.5, 1)),
    ("binary", "min_data_in_leaf_ratio", _float(0.01, 0.2)),
    ("binary", "min_data_in_leaf_ratio", _cat(None, 0.05)),
    ("binary", "min_data_in_bin_ratio", _float(0.01, 0.2)),
    ("binary", "min_data_in_bin_ratio", _cat(None, 0.1)),
]


def _accepted_id(row: tuple[str, str, dict[str, Any]]) -> str:
    task, name, spec = row
    shown = (
        repr(spec["choices"])
        if spec["type"] == "categorical"
        else f"{spec['type']}({spec['low']},{spec['high']})"
    )
    return f"{task}-{name}-{shown}"


@pytest.mark.parametrize("row", ACCEPTED, ids=[_accepted_id(r) for r in ACCEPTED])
def test_accepted_values_reach_training_as_chosen(
    row: tuple[str, str, dict[str, Any]], smart_seen: list[dict[str, Any]]
) -> None:
    task, name, spec = row
    cfg = _config(task, {name: spec}, n_trials=3)
    result = Model(cfg, data=_frame(task)).tune()
    trained = [smart[name] for smart in smart_seen]
    assert trained, "no trial resolved its smart parameters"
    for value in [*trained, result.best_smart_params[name]]:
        if spec["type"] == "categorical":
            assert any(_same(value, c) for c in spec["choices"]), value
        else:
            assert type(value) is float and spec["low"] <= value <= spec["high"]


def test_the_347_reproduction_false_trains_without_num_leaves() -> None:
    """``[False]`` keeps native leaves; ``["false"]`` is refused (criterion 1)."""
    cfg = _config("binary", {"auto_num_leaves": _cat(False)})
    with record_lightgbm_calls() as seen:
        result = Model(cfg, data=_frame("binary")).tune()
    assert result.best_smart_params == {"auto_num_leaves": False}
    assert seen["train_params"]
    assert all("num_leaves" not in params for params in seen["train_params"])


# ---------------------------------------------------------------------------
# Criterion 3: the num_leaves_ratio expansion stops at 1.0
# ---------------------------------------------------------------------------


def _steer_ratio_to(monkeypatch: pytest.MonkeyPatch, value: float) -> None:
    """Round 1's first trial samples *value* and is the best one."""
    real_run = Tuner.run
    real_build = Model._build_tune_objective

    def run(self: Tuner, study: Any, objective: Any, **kwargs: Any) -> None:
        if kwargs["round_number"] == 1:
            study.enqueue_trial({"num_leaves_ratio": value})
        real_run(self, study, objective, **kwargs)

    def build(self: Model, **kwargs: Any) -> Callable[[Any], float]:
        objective = real_build(self, **kwargs)

        def steered(trial: Any) -> float:
            objective(trial)
            return 0.0 if trial.params.get("num_leaves_ratio") == value else 1.0

        return steered

    monkeypatch.setattr(Tuner, "run", run)
    monkeypatch.setattr(Model, "_build_tune_objective", build)


def _merge_config(n_trials: int = 2) -> dict[str, Any]:
    """The provider's default space, with cheap trees (merge mode)."""
    return _config(
        "regression", {"n_estimators": SMALL_TREES}, mode="merge", n_trials=n_trials
    )


def test_default_ratio_best_at_one_is_clamped_not_expanded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    storage = url(tmp_path)
    _steer_ratio_to(monkeypatch, 1.0)
    cfg = _merge_config()
    model = Model(cfg, data=_frame("regression"))
    model.tune(storage=storage, study_name="s", expand_boundary=True)
    first = round_record(storage, 1)
    (ratio_1,) = [d for d in first.effective if d.name == "num_leaves_ratio"]
    assert (ratio_1.low, ratio_1.high) == (0.5, 1.0)
    model.tune(resume=True, storage=storage, study_name="s", expand_boundary=True)
    second = round_record(storage, 2)
    (ratio_2,) = [d for d in second.effective if d.name == "num_leaves_ratio"]
    assert isinstance(ratio_2, FloatDim)
    assert ratio_2.high <= 1.0
    assert second.boundary_report is not None
    (status,) = [s for s in second.boundary_report.dims if s.name == "num_leaves_ratio"]
    assert status.edge == "upper"
    assert status.expanded is False
    assert status.clamped_to_bound is True


# ---------------------------------------------------------------------------
# Criterion 4: every source of a round's space is checked before its record
# ---------------------------------------------------------------------------


def test_an_expanded_dimension_outside_the_rule_is_refused_before_the_record(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The expansion swapped for one that ignores the provider's bound."""
    storage = url(tmp_path)
    cfg = _merge_config()
    model = Model(cfg, data=_frame("regression"))
    model.tune(storage=storage, study_name="s")
    real = _tuning_plan.expand_for_next_round

    def overshoot(
        previous: Any, *args: Any, **kwargs: Any
    ) -> tuple[tuple[SearchDim, ...], Any]:
        effective, report = real(previous, *args, **kwargs)
        return (
            tuple(
                dataclasses.replace(d, high=1.5, max_allowed=None)
                if d.name == "num_leaves_ratio"
                else d
                for d in effective
            ),
            report,
        )

    monkeypatch.setattr(_tuning_plan, "expand_for_next_round", overshoot)
    before = (snapshot_study(storage, "s"), snapshot_model(model))
    with record_lightgbm_calls() as seen, pytest.raises(LizyMLError) as caught:
        model.tune(resume=True, storage=storage, study_name="s", expand_boundary=True)
    _assert_refusal(
        caught.value,
        surface="tuning.optuna.space",
        name="num_leaves_ratio",
        key="high",
        value=1.5,
        reason=INVALID,
    )
    assert seen["train_params"] == []
    assert (snapshot_study(storage, "s"), snapshot_model(model)) == before


class _BadDefaultProvider(LGBMProvider):
    """The shipped provider whose default ``num_leaves_ratio`` reaches 2.0."""

    def default_space(self, task: Any) -> list[SearchDim]:
        return [
            dataclasses.replace(d, high=2.0) if d.name == "num_leaves_ratio" else d
            for d in super().default_space(task)
        ]


def test_a_default_space_dimension_outside_the_rule_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(_model_tuning, "get_provider", lambda _c: _BadDefaultProvider())
    storage = url(tmp_path)
    with record_lightgbm_calls() as seen, pytest.raises(LizyMLError) as caught:
        Model(_merge_config(), data=_frame("regression")).tune(
            storage=storage, study_name="s"
        )
    _assert_refusal(
        caught.value,
        surface="tuning.optuna.space",
        name="num_leaves_ratio",
        key="high",
        value=2.0,
        reason=INVALID,
    )
    assert seen["train_params"] == []
    assert [
        s.study_name
        for s in optuna.get_all_study_summaries(storage, include_best_trial=False)
    ] == []


def test_the_rule_precedes_the_owner_check() -> None:
    """``[0]`` is refused for its type, not read as an owner that is never true.

    The H-0121 owner check reads an owner dimension by truthiness; without the
    rule before it, ``0`` would make it refuse the ratio dimension instead.
    """
    space = {"auto_num_leaves": _cat(0), "num_leaves_ratio": _float(0.5, 1.0)}
    with pytest.raises(LizyMLError) as caught:
        Model(_config("binary", space), data=_frame("binary")).tune()
    _assert_refusal(
        caught.value,
        surface="tuning.optuna.space",
        name="auto_num_leaves",
        key="choices",
        value=0,
        reason=INVALID,
    )


def test_the_rule_precedes_the_owner_check_on_a_space_arg(
    stored_studies: dict[str, Path], tmp_path: Path
) -> None:
    shutil.copy(stored_studies["binary"], tmp_path / "s.db")
    storage = url(tmp_path)
    model = Model(_config("binary", {"learning_rate": LR}), data=_frame("binary"))
    space = {
        "learning_rate": LR,
        "auto_num_leaves": _cat(0),
        "num_leaves_ratio": _float(0.5, 1.0),
    }
    with pytest.raises(LizyMLError) as caught:
        model.tune(resume=True, storage=storage, study_name="s", space=space)
    _assert_refusal(
        caught.value,
        surface="tune(space=)",
        name="auto_num_leaves",
        key="choices",
        value=0,
        reason=INVALID,
    )


# ---------------------------------------------------------------------------
# Criterion 6: restore_tuning
# ---------------------------------------------------------------------------


def test_restore_tuning_refuses_a_string_bool_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage = url(tmp_path)
    cfg = _config("binary", {"auto_num_leaves": _cat("false")})
    # The study is made with the rule before this amendment: no dimension
    # value checked (H-0122 lineages made on develop before it can hold one).
    with monkeypatch.context() as mp:
        from lizyml.core import _smart_values

        mp.setattr(_smart_values, "check_smart_dimension_values", lambda *a, **k: None)
        Model(cfg, data=_frame("binary")).tune(storage=storage, study_name="s")
    other = Model(cfg, data=_frame("binary"))
    with pytest.raises(LizyMLError) as caught:
        other.restore_tuning(storage, "s")
    err = caught.value
    assert err.code is CI
    assert err.context["reason"] == INVALID
    assert err.context["name"] == "auto_num_leaves"
    assert err.context["value"] == "false"
    assert other._tuning_result is None


@pytest.mark.parametrize("variant", ["reattach", "in_memory"])
def test_resume_refuses_a_legacy_best_whose_smart_dimension_was_removed(
    variant: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The study-wide best is checked before any write, not only the new round.

    A ``space=`` round without the smart dimension keeps the legacy best (a
    round-1 trial recording ``"false"``) as the tune's best; it would reach
    the result, the ``fit`` overlay and the export.
    """
    storage = url(tmp_path) if variant == "reattach" else None
    cfg = _config("binary", {"auto_num_leaves": _cat("false"), "learning_rate": LR})
    with monkeypatch.context() as mp:
        from lizyml.core import _smart_values

        mp.setattr(_smart_values, "check_smart_dimension_values", lambda *a, **k: None)
        first = Model(cfg, data=_frame("binary"))
        first.tune(storage=storage, study_name="s" if storage else None)
    model = Model(cfg, data=_frame("binary")) if variant == "reattach" else first
    study = first._study
    before = (snapshot_study(study), snapshot_model(model))
    with pytest.raises(LizyMLError) as caught:
        model.tune(
            resume=True,
            storage=storage,
            study_name="s" if storage else None,
            space={"learning_rate": LR},
        )
    err = caught.value
    assert err.code is CI
    assert (err.context["name"], err.context["value"]) == ("auto_num_leaves", "false")
    assert err.context["reason"] == INVALID
    assert (snapshot_study(study), snapshot_model(model)) == before


@pytest.mark.parametrize("surface", ["resume", "restore"])
def test_a_legacy_non_best_trial_outside_the_rule_is_refused(
    surface: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every stored trial is checked, not only the best.

    The legacy round tried ``False`` (the best) and ``"false"``; the string
    would reach ``TuningResult.trials`` and the tables even with the valid
    best, so the study is refused on resume and on restore.
    """
    from tests.test_tuning.test_search_dim_step_model import steer

    storage = url(tmp_path)
    cfg = _config(
        "regression",
        {"auto_num_leaves": _cat(False, "false"), "learning_rate": LR},
        n_trials=2,
    )
    with monkeypatch.context() as mp:
        from lizyml.core import _smart_values

        mp.setattr(_smart_values, "check_smart_dimension_values", lambda *a, **k: None)
        steer(
            mp,
            {
                1: [
                    {"auto_num_leaves": False, "learning_rate": 0.1},
                    {"auto_num_leaves": "false", "learning_rate": 0.1},
                ]
            },
            lambda params: params["auto_num_leaves"] is False,
        )
        Model(cfg, data=_frame("regression")).tune(storage=storage, study_name="s")
    study = optuna.load_study(study_name="s", storage=storage)
    assert study.best_trial.params["auto_num_leaves"] is False
    model = Model(cfg, data=_frame("regression"))
    before = snapshot_study(study)
    with pytest.raises(LizyMLError) as caught:
        if surface == "resume":
            model.tune(
                resume=True,
                storage=storage,
                study_name="s",
                space={"learning_rate": LR},
            )
        else:
            model.restore_tuning(storage, "s")
    err = caught.value
    assert err.code is CI
    assert (err.context["name"], err.context["value"]) == ("auto_num_leaves", "false")
    assert "trial 1" in err.user_message
    assert snapshot_study(optuna.load_study(study_name="s", storage=storage)) == before
    assert model._tuning_result is None


def test_restore_tuning_accepts_a_result_inside_the_rule(tmp_path: Path) -> None:
    storage = url(tmp_path)
    cfg = _config("binary", {"auto_num_leaves": _cat(True, False)})
    Model(cfg, data=_frame("binary")).tune(storage=storage, study_name="s")
    restored = Model(cfg, data=_frame("binary")).restore_tuning(storage, "s")
    assert type(restored.best_smart_params["auto_num_leaves"]) is bool
