"""H-0128 (#348, #349): a round with no COMPLETE trial of its own fails, LightGBM
errors are trial failures, and an open round with no COMPLETE trial can be
replaced by a new round with a different ``space=``.

Each test names the HISTORY.md H-0128 acceptance criterion it pins.
"""

from __future__ import annotations

import copy
import logging
import math
import warnings
from typing import Any

import optuna
import pytest

from lizyml import Model
from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.core.types.search_dim import FloatDim, SearchDim
from lizyml.estimators.lgbm.provider import LGBMProvider
from lizyml.estimators.provider import EstimatorProvider, TrialErrorProvider
from lizyml.tuning import tuner as tuner_module
from lizyml.tuning.lineage import classify_study, end_key, round_key
from lizyml.tuning.study_lineage import read_study_lineage
from lizyml.tuning.trial_failure import MAX_TEXT, describe_exception
from lizyml.tuning.tuner import Tuner
from tests.test_tuning._resume_helpers import (
    SPACE,
    ProcessDeath,
    die_after_attr,
    die_at_attr,
    die_in_objective,
    fast_config,
    frame,
    load,
    round_record,
    snapshot_study,
    url,
)

optuna.logging.set_verbosity(optuna.logging.ERROR)

#: A space every trial of which LightGBM refuses (``max_bin`` must exceed 1).
BAD_BIN: dict[str, Any] = {
    **copy.deepcopy(SPACE),
    "max_bin": {"type": "int", "low": 0, "high": 1},
}
#: A corrected space for the round after a failed one.
GOOD: dict[str, Any] = copy.deepcopy(SPACE)
#: A space that differs from the config's (another learning-rate range).
OTHER: dict[str, Any] = {
    **copy.deepcopy(SPACE),
    "learning_rate": {"type": "float", "low": 0.1, "high": 0.2, "log": True},
}


def _fail_trials(monkeypatch: pytest.MonkeyPatch, model: Model, exc: Exception) -> None:
    """Make every trial of *model*'s next tune raise *exc* inside the objective."""

    def build(*_: Any, **__: Any) -> Any:
        raise exc

    monkeypatch.setattr(model, "_build_train_components", build)


def _lineage(storage: str, name: str = "s") -> Any:
    study = load(storage, name)
    return classify_study(study.user_attrs, study.trials)


def _end_status(storage: str, k: int, name: str = "s") -> str | None:
    for record in read_study_lineage(storage, name).rounds:
        if record.round == k:
            return str(record.status)
    return None


def _states(n: int, fail: int, pruned: int = 0, interrupted: int = 0) -> dict[str, int]:
    return {"trials": n, "FAIL": fail, "PRUNED": pruned, "interrupted": interrupted}


# --- Criterion 1 (#348) -------------------------------------------------------


def test_a_resumed_round_with_no_complete_trial_fails(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage = url(tmp_path)
    model = Model(fast_config(), data=frame())
    first = model.tune(storage=storage, study_name="s")
    with monkeypatch.context() as mp:
        _fail_trials(mp, model, ValueError("bad range"))
        with pytest.raises(LizyMLError) as caught:
            model.tune(resume=True)
    err = caught.value
    assert err.code is ErrorCode.TUNING_FAILED
    assert err.context["round"] == 2 and err.context["status"] == "failed"
    assert err.context["trial_states"] == _states(2, 2)
    assert err.context["first_error"] == "ValueError: bad range"
    assert "earlier rounds" in err.user_message
    assert _end_status(storage, 2) == "failed"
    # The Model is unchanged: its tuning result is still round 1's.
    assert [r.round for r in model._tuning_result.rounds] == [1]
    assert model._tuning_result.best_score == first.best_score
    # A corrected space= starts round 3 from the study's best.
    third = model.tune(resume=True, space=GOOD)
    assert [r.round for r in third.rounds] == [1, 3]
    assert third.rounds[-1].best_score_before == first.best_score


# --- Criterion 2 --------------------------------------------------------------


def test_a_round_with_some_complete_trials_stays_completed(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage = url(tmp_path)
    model = Model(fast_config(), data=frame())
    model.tune(storage=storage, study_name="s")
    real = model._build_train_components
    calls = {"n": 0}

    def first_fails(*args: Any, **kwargs: Any) -> Any:
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError("only the first")
        return real(*args, **kwargs)

    monkeypatch.setattr(model, "_build_train_components", first_fails)
    model.tune(resume=True)
    assert _end_status(storage, 2) == "completed"


# --- Criterion 3: both special close paths, after a completed round 1 --------


def _open_round_2(
    monkeypatch: pytest.MonkeyPatch, storage: str, n_trials: int
) -> Model:
    """Round 1 completed; round 2 (``n_trials``) dies in its first trial,
    which stays RUNNING: an open round with no COMPLETE trial."""
    model = Model(fast_config(n_trials=n_trials), data=frame())
    model.tune(n_trials=2, storage=storage, study_name="s")
    with monkeypatch.context() as mp:
        die_in_objective(mp, model, at_call=1)
        with pytest.raises(ProcessDeath):
            model.tune(resume=True)
    assert _lineage(storage).state == "open"
    return model


def test_finishing_an_open_round_without_a_complete_trial_fails(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage = url(tmp_path)
    _open_round_2(monkeypatch, storage, n_trials=2)
    model = Model(fast_config(n_trials=2), data=frame())
    with monkeypatch.context() as mp:
        _fail_trials(mp, model, ValueError("still bad"))
        with pytest.raises(LizyMLError) as caught:
            model.tune(resume=True, storage=storage, study_name="s")
    assert caught.value.code is ErrorCode.TUNING_FAILED
    assert caught.value.context["trial_states"] == _states(2, 2, interrupted=1)
    assert _end_status(storage, 2) == "failed"  # the old rule closed it completed


def test_an_open_round_with_no_budget_left_and_no_complete_trial_fails(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage = url(tmp_path)
    _open_round_2(monkeypatch, storage, n_trials=1)
    with pytest.raises(LizyMLError) as caught:
        Model(fast_config(n_trials=1), data=frame()).tune(
            resume=True, storage=storage, study_name="s"
        )
    assert caught.value.code is ErrorCode.TUNING_FAILED
    assert caught.value.context["trial_states"] == _states(1, 1, interrupted=1)
    assert "first_error" not in caught.value.context
    assert _end_status(storage, 2) == "failed"


# --- Criterion 4: the public Tuner.tune --------------------------------------

_DIMS: list[SearchDim] = [FloatDim(name="x", low=0.0, high=1.0)]


def _stored_study(tmp_path: Any) -> tuple[str, Any]:
    storage = url(tmp_path, "t.db")
    tuner = Tuner(_DIMS, n_trials=2, storage=storage, study_name="t")
    tuner.tune(lambda t: t.suggest_float("x", 0.0, 1.0))
    return storage, load(storage, "t")


def test_public_tuner_judges_only_its_own_trials(tmp_path: Any) -> None:
    storage, _ = _stored_study(tmp_path)

    def fail(trial: Any) -> float:
        trial.suggest_float("x", 0.0, 1.0)
        raise ValueError("nope")

    with pytest.raises(LizyMLError) as caught:
        Tuner(_DIMS, n_trials=2, storage=storage, study_name="t").tune(fail)
    assert caught.value.code is ErrorCode.TUNING_FAILED
    assert caught.value.context["trial_states"] == _states(2, 2)


def test_an_enqueued_trial_belongs_to_the_call(tmp_path: Any) -> None:
    storage, study = _stored_study(tmp_path)
    boundary = len(study.trials)

    def only_enqueued_succeeds(trial: Any) -> float:
        x = trial.suggest_float("x", 0.0, 1.0)
        if trial.number != boundary:
            raise ValueError("sampled trial")
        return x

    tuner = Tuner(_DIMS, n_trials=2, storage=storage, study_name="t")
    tuner.tune(only_enqueued_succeeds, enqueue_params={"x": 0.5})


# --- Criterion 5 (#349 (1)): real LightGBM errors -----------------------------


def test_a_lightgbm_error_fails_the_trial_not_the_round(
    tmp_path: Any, caplog: pytest.LogCaptureFixture
) -> None:
    storage = url(tmp_path)
    model = Model(fast_config(n_trials=2), data=frame())
    model.tune(storage=storage, study_name="s")
    with caplog.at_level(logging.WARNING), pytest.raises(LizyMLError) as caught:
        model.tune(resume=True, space=BAD_BIN)
    err = caught.value
    assert err.code is ErrorCode.TUNING_FAILED and err.context["status"] == "failed"
    assert err.context["trial_states"] == _states(2, 2)
    assert "max_bin" in err.context["first_error"] and "max_bin" in err.user_message
    logged = [
        r.getMessage() for r in caplog.records if "trial.failed" in r.getMessage()
    ]
    assert len(logged) == 2 and all("max_bin" in m for m in logged)
    third = model.tune(resume=True, space=GOOD)
    assert [r.round for r in third.rounds] == [1, 3]


def test_a_lightgbm_error_in_round_1_fails_the_round(tmp_path: Any) -> None:
    cfg = fast_config(n_trials=2)
    cfg["tuning"]["optuna"]["space"] = copy.deepcopy(BAD_BIN)
    with pytest.raises(LizyMLError) as caught:
        Model(cfg, data=frame()).tune(storage=url(tmp_path), study_name="s")
    assert caught.value.context["trial_states"] == _states(2, 2)
    assert "max_bin" in caught.value.context["first_error"]


# --- Criterion 5b: the failure report ------------------------------------------


class _CustomError(Exception):
    pass


def _run(
    objective: Any, *, n_trials: int = 1, extra_catch: tuple[type[Exception], ...] = ()
) -> Tuner:
    tuner = Tuner(_DIMS, n_trials=n_trials)
    study = optuna.create_study()
    tuner.run(study, objective, round_number=1, round_start=0, extra_catch=extra_catch)
    tuner.study_for_test = study  # type: ignore[attr-defined]
    return tuner


@pytest.mark.parametrize(
    ("exc", "extra"),
    [
        (LizyMLError(ErrorCode.CONFIG_INVALID, user_message="lizy"), ()),
        (ValueError("value"), ()),
        (RuntimeError("runtime"), ()),
        (_CustomError("custom"), (_CustomError,)),
    ],
    ids=["LizyMLError", "ValueError", "RuntimeError", "extra_catch"],
)
def test_each_caught_family_is_a_logged_trial_failure(
    exc: Exception, extra: tuple[type[Exception], ...], caplog: pytest.LogCaptureFixture
) -> None:
    def objective(trial: Any) -> float:
        trial.suggest_float("x", 0.0, 1.0)
        raise exc

    with caplog.at_level(logging.WARNING):
        tuner = _run(objective, extra_catch=extra)
    study = tuner.study_for_test  # type: ignore[attr-defined]
    assert [t.state.name for t in study.trials] == ["FAIL"]
    assert tuner.first_error == describe_exception(exc)
    logged = [
        r.getMessage() for r in caplog.records if "trial.failed" in r.getMessage()
    ]
    assert logged and type(exc).__name__ in logged[0]
    assert "reason=unknown" not in logged[0]  # what the log said before H-0128
    assert study.trials[0].user_attrs == {}  # nothing stored on the trial


def test_an_uncaught_exception_reports_and_leaves_the_round_open(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    storage = url(tmp_path)
    model = Model(fast_config(n_trials=2), data=frame())
    model.tune(storage=storage, study_name="s")
    with monkeypatch.context() as mp:
        _fail_trials(mp, model, TypeError("not caught"))
        with caplog.at_level(logging.WARNING), pytest.raises(LizyMLError) as caught:
            model.tune(resume=True)
    err = caught.value
    assert err.user_message.startswith("Optuna study failed")
    logged = [
        r.getMessage() for r in caplog.records if "trial.failed" in r.getMessage()
    ]
    assert len(logged) == 1 and "TypeError: not caught" in logged[0]
    assert err.context["first_error"] == "TypeError: not caught"
    assert err.context["trial_states"] == _states(1, 1)
    assert err.context["round"] == 2
    assert _lineage(storage).state == "open"


def test_an_uncaught_unprintable_exception_still_raises_tuning_failed() -> None:
    class Hostile(TypeError):
        def __str__(self) -> str:
            raise OSError("no str")

    def objective(trial: Any) -> float:
        trial.suggest_float("x", 0.0, 1.0)
        raise Hostile()

    with pytest.raises(LizyMLError) as caught:  # not the OSError
        _run(objective)
    assert caught.value.user_message == "Optuna study failed: <unprintable message>"
    assert caught.value.context["first_error"] == "Hostile: <unprintable message>"


def test_without_extra_catch_a_lightgbm_error_stops_the_public_tuner() -> None:
    from lightgbm.basic import LightGBMError

    def objective(trial: Any) -> float:
        trial.suggest_float("x", 0.0, 1.0)
        raise LightGBMError("Check failed")

    with pytest.raises(LizyMLError) as caught:
        _run(objective)
    assert caught.value.user_message.startswith("Optuna study failed")
    assert caught.value.context["first_error"].startswith("LightGBMError")


def test_the_first_error_is_kept_not_the_last() -> None:
    errors = iter([ValueError("first"), RuntimeError("second")])

    def objective(trial: Any) -> float:
        trial.suggest_float("x", 0.0, 1.0)
        raise next(errors)

    assert _run(objective, n_trials=2).first_error == "ValueError: first"


def test_long_texts_are_cut() -> None:
    long_type = type("E" * (MAX_TEXT + 100), (ValueError,), {})
    text = describe_exception(long_type("m" * (MAX_TEXT + 100)))
    name, message = text.split(": ", 1)
    assert len(name) == MAX_TEXT and len(message) == MAX_TEXT


def test_an_unprintable_message_does_not_stop_the_round() -> None:
    class Hostile(ValueError):
        def __str__(self) -> str:
            raise OSError("no str")

    def objective(trial: Any) -> float:
        trial.suggest_float("x", 0.0, 1.0)
        raise Hostile()

    tuner = _run(objective, n_trials=2)
    states = [t.state.name for t in tuner.study_for_test.trials]  # type: ignore[attr-defined]
    assert states == ["FAIL", "FAIL"]  # the round went on
    assert tuner.first_error == "Hostile: <unprintable message>"


def test_rounds_without_an_exception_report_counts_only() -> None:
    def nan(trial: Any) -> float:
        trial.suggest_float("x", 0.0, 1.0)
        return math.nan

    def pruned(trial: Any) -> float:
        trial.suggest_float("x", 0.0, 1.0)
        raise optuna.TrialPruned()

    for objective, expected in (
        (nan, _states(2, 2)),
        (pruned, _states(2, 0, pruned=2)),
    ):
        with pytest.raises(LizyMLError) as caught:
            Tuner(_DIMS, n_trials=2).tune(objective)
        assert caught.value.context["trial_states"] == expected
        assert "first_error" not in caught.value.context


def test_a_zero_trial_round_reports_its_budget(tmp_path: Any) -> None:
    cfg = fast_config(n_trials=2)
    cfg["tuning"]["optuna"]["params"]["timeout"] = 0
    with pytest.raises(LizyMLError) as caught:
        Model(cfg, data=frame()).tune(storage=url(tmp_path), study_name="s")
    assert caught.value.context["trial_states"] == _states(0, 0)
    assert "no trial ran" in caught.value.user_message


def test_public_tuner_tune_forwards_extra_catch() -> None:
    def objective(trial: Any) -> float:
        trial.suggest_float("x", 0.0, 1.0)
        raise _CustomError("custom")

    with pytest.raises(LizyMLError) as caught:
        Tuner(_DIMS, n_trials=2).tune(objective, extra_catch=(_CustomError,))
    # Recorded as FAIL and the call went on: the all-failed report, not
    # "Optuna study failed" from the first trial.
    assert caught.value.context["trial_states"] == _states(2, 2)


# --- Criterion 5c: providers without the optional capability -----------------


def test_trial_errors_is_a_separate_optional_capability() -> None:
    assert "trial_errors" not in dir(EstimatorProvider)
    assert isinstance(LGBMProvider(), TrialErrorProvider)


def test_a_provider_without_trial_errors_tunes_as_before(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    import lizyml.core._model_tuning as model_tuning

    class ThirdParty:
        """An LGBM provider seen through a pre-H-0128 interface."""

        def __init__(self) -> None:
            self._inner = LGBMProvider()

        def __getattr__(self, name: str) -> Any:
            if name == "trial_errors":
                raise AttributeError(name)
            return getattr(self._inner, name)

    monkeypatch.setattr(model_tuning, "get_provider", lambda cfg: ThirdParty())
    model = Model(fast_config(n_trials=2), data=frame())
    model.tune(storage=url(tmp_path), study_name="s")  # tunes as before
    with pytest.raises(LizyMLError) as caught:
        model.tune(resume=True, space=BAD_BIN)
    # No extra_catch: the LightGBM error ends the round as before H-0128.
    assert caught.value.user_message.startswith("Optuna study failed")


# --- Criterion 6 (#349 (2)): replacing an open round ---------------------------


def test_a_different_space_replaces_an_open_round_without_complete_trials(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage = url(tmp_path)
    _open_round_2(monkeypatch, storage, n_trials=2)
    model = Model(fast_config(n_trials=2), data=frame())
    with pytest.warns(UserWarning, match="Round 2 had no completed trial"):
        result = model.tune(
            resume=True,
            storage=storage,
            study_name="s",
            space=OTHER,
            n_trials=3,
            expand_boundary=True,
        )
    assert _end_status(storage, 2) == "failed"
    assert _end_status(storage, 3) == "completed"
    assert [r.round for r in result.rounds] == [1, 3]
    assert result.rounds[-1].n_trials == 3  # the replacement's own n_trials
    # ... and its own expand_boundary, which U2 would refuse on the open round.
    assert round_record(storage, 2).expand_boundary is False
    assert round_record(storage, 3).expand_boundary is True


@pytest.mark.parametrize("space", [None, GOOD], ids=["no space", "same space"])
def test_continuing_an_open_round_is_unchanged(
    space: dict[str, Any] | None, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage = url(tmp_path)
    _open_round_2(monkeypatch, storage, n_trials=2)
    result = Model(fast_config(n_trials=2), data=frame()).tune(
        resume=True, storage=storage, study_name="s", space=space
    )
    assert [r.round for r in result.rounds] == [1, 2]


def test_an_open_round_with_a_complete_trial_keeps_u2(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage = url(tmp_path)
    model = Model(fast_config(n_trials=3), data=frame())
    model.tune(n_trials=2, storage=storage, study_name="s")
    with monkeypatch.context() as mp:
        die_in_objective(mp, model, at_call=2)  # trial 1 COMPLETE, trial 2 RUNNING
        with pytest.raises(ProcessDeath):
            model.tune(resume=True)
    with pytest.raises(LizyMLError) as caught:
        Model(fast_config(n_trials=3), data=frame()).tune(
            resume=True, storage=storage, study_name="s", space=OTHER
        )
    assert caught.value.code is ErrorCode.CONFIG_INVALID


def test_a_zero_trial_open_round_can_be_replaced(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage = url(tmp_path)
    model = Model(fast_config(n_trials=2), data=frame())
    model.tune(storage=storage, study_name="s")
    with monkeypatch.context() as mp:
        die_after_attr(mp, round_key(2))
        with pytest.raises(ProcessDeath):
            model.tune(resume=True)
    with pytest.warns(UserWarning, match="Round 2"):
        Model(fast_config(n_trials=2), data=frame()).tune(
            resume=True, storage=storage, study_name="s", space=OTHER
        )
    assert _end_status(storage, 2) == "failed"
    assert _end_status(storage, 3) == "completed"


# --- Criterion 6b: dying inside the replacement ---------------------------------


def _die_mid_fail_running(monkeypatch: pytest.MonkeyPatch) -> None:
    real = tuner_module.fail_running_trials

    def fail_one_then_die(study: Any, numbers: Any) -> None:
        real(study, tuple(numbers)[:1])
        raise ProcessDeath("died while failing RUNNING trials")

    monkeypatch.setattr(tuner_module, "fail_running_trials", fail_one_then_die)


@pytest.mark.parametrize(
    "window", ["during (a)", "after (a)", "after (b)", "after (c)"]
)
def test_dying_inside_the_replacement_recovers(
    window: str, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage = url(tmp_path)
    _open_round_2(monkeypatch, storage, n_trials=2)
    load(storage, "s").ask()  # a second RUNNING trial in round 2
    model = Model(fast_config(n_trials=2), data=frame())
    with monkeypatch.context() as mp:
        if window == "during (a)":
            _die_mid_fail_running(mp)
        elif window == "after (a)":
            die_at_attr(mp, end_key(2))
        elif window == "after (b)":
            die_at_attr(mp, round_key(3))
        else:
            die_in_objective(mp, model, at_call=1)
        with pytest.raises(ProcessDeath), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.tune(resume=True, storage=storage, study_name="s", space=OTHER)
    expected = {"during (a)": "open", "after (a)": "open", "after (b)": "failed"}
    assert _lineage(storage).state == expected.get(window, "open")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = Model(fast_config(n_trials=2), data=frame()).tune(
            resume=True, storage=storage, study_name="s", space=OTHER
        )
    assert [r.round for r in result.rounds] == [1, 3]
    assert _end_status(storage, 2) == "failed"


# --- Criterion 7: a refusal inside the replacement path writes nothing --------


def test_a_refusal_after_replacement_planning_writes_nothing(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage = url(tmp_path)
    cfg = fast_config(n_trials=2)
    cfg["model"]["auto_num_leaves"] = True
    del cfg["tuning"]["optuna"]["space"]["num_leaves"]
    model = Model(cfg, data=frame())
    model.tune(storage=storage, study_name="s")
    with monkeypatch.context() as mp:
        die_in_objective(mp, model, at_call=1)
        with pytest.raises(ProcessDeath):
            model.tune(resume=True)
    before = snapshot_study(storage, "s")
    assert "RUNNING" in [t[1] for t in before[0]]
    smart_managed = {
        **cfg["tuning"]["optuna"]["space"],
        "num_leaves": SPACE["num_leaves"],
    }
    with pytest.raises(LizyMLError) as caught:
        Model(cfg, data=frame()).tune(
            resume=True, storage=storage, study_name="s", space=smart_managed
        )
    assert caught.value.code is ErrorCode.CONFIG_INVALID
    assert snapshot_study(storage, "s") == before


# --- Criterion 8 ---------------------------------------------------------------


def test_restore_refuses_a_failed_last_round_after_a_completed_one(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage = url(tmp_path)
    model = Model(fast_config(), data=frame())
    model.tune(storage=storage, study_name="s")
    with monkeypatch.context() as mp:
        _fail_trials(mp, model, ValueError("bad range"))
        with pytest.raises(LizyMLError):
            model.tune(resume=True)
    assert read_study_lineage(storage, "s").state == "failed"
    with pytest.raises(LizyMLError) as caught:
        Model(fast_config(), data=frame()).restore_tuning(
            storage=storage, study_name="s"
        )
    message = caught.value.user_message
    assert "(2)" in message
    assert "no trial of that round completed" in message
    assert "earlier rounds are not restored" in message
