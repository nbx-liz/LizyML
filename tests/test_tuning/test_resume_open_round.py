"""H-0122 §6.5, §7.1, §7.2, §7.4 state O: a round a crash left open.

Criteria 8 (a crash mid-round is finished on resume, in-process, by a new
Model, and after a real process kill; a crash before ``:end`` closes the round
without trials), 8a (a round whose only trial was left RUNNING closes as
``failed``), 8b without ``space=`` (U2: ``n_trials`` / ``expand_boundary`` are
compared with the record), 16 (the remaining timeout), 18 (progress continues
at K+1) and 31 (``RoundSummary.n_trials`` is the actual count).

The scenario (``_resume_helpers.crash_round_2``): round 1 runs 2 trials, round
2 asks for 3 with ``expand_boundary=True``, and the process dies in round 2's
2nd trial (#3), which stays RUNNING. Sampled values are never compared across
a crashed and an uninterrupted run: a reattach builds a fresh sampler (design
R11). Counts, numbering, rounds and the boundary report are.
"""

from __future__ import annotations

import copy
import logging
import shutil
from pathlib import Path
from typing import Any

import pytest

from lizyml import Model
from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.core.types.tuning_result import TuneProgressInfo
from lizyml.tuning.lineage import classify_study, end_key, round_key
from lizyml.tuning.lineage_codec import decode_round_end
from tests.test_tuning._resume_helpers import (
    CRASHED_TRIAL,
    ROUND_1_TRIALS,
    ROUND_2_TRIALS,
    ProcessDeath,
    crash_round_2,
    die_in_objective,
    edge_config,
    fast_config,
    frame,
    load,
    run_child_and_kill,
    snapshot_model,
    snapshot_study,
    spy_tuner,
    sqlite_exec,
    url,
)

TOTAL = ROUND_1_TRIALS + ROUND_2_TRIALS


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def reference() -> Any:
    """The uninterrupted run: round 1 and round 2, no crash."""
    model = Model(edge_config(), data=frame())
    model.tune(n_trials=ROUND_1_TRIALS)
    return model.tune(resume=True, expand_boundary=True)


@pytest.fixture(scope="module")
def open_template(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A stored study in state O: round 2 died with trial #3 RUNNING."""
    root = tmp_path_factory.mktemp("open_template")
    with pytest.MonkeyPatch.context() as mp:
        crash_round_2(mp, Model(edge_config(), data=frame()), url(root), "s")
    return root / "s.db"


@pytest.fixture
def storage(open_template: Path, tmp_path: Path) -> str:
    shutil.copy(open_template, tmp_path / "s.db")
    return url(tmp_path)


def _assert_round_2_open(study: Any) -> None:
    """The crash residue: round 2 recorded, not ended, its 2nd trial RUNNING."""
    assert round_key(2) in study.user_attrs
    assert end_key(2) not in study.user_attrs
    states = {t.number: t.state.name for t in study.trials}
    assert states == {0: "COMPLETE", 1: "COMPLETE", 2: "COMPLETE", 3: "RUNNING"}
    assert classify_study(study.user_attrs, study.trials).state == "open"


def _round_2_record(study: Any) -> Any:
    lineage = classify_study(study.user_attrs, study.trials)
    return next(r for r in lineage.rounds if r.round == 2)


def _assert_round_2_finished(result: Any, study: Any, reference: Any) -> None:
    """Round 2 ended with its budget, matching the uninterrupted run (INV-R5)."""
    assert len(study.trials) == TOTAL
    assert study.trials[CRASHED_TRIAL].state.name == "FAIL"
    end = decode_round_end(study.user_attrs[end_key(2)])
    assert end.status == "completed"
    assert [(r.round, r.n_trials) for r in result.rounds] == [
        (r.round, r.n_trials) for r in reference.rounds
    ]
    assert [t.round for t in result.trials] == [t.round for t in reference.trials]
    assert result.trials[CRASHED_TRIAL].state == "fail"
    record = _round_2_record(study)
    # The report is the one round 2 decided when it started; round 1 ran
    # uninterrupted with the same seed in both runs, so it is also the
    # uninterrupted run's.
    assert record.boundary_report is not None
    assert record.boundary_report.expanded_names
    assert result.boundary_report == record.boundary_report
    assert result.boundary_report == reference.boundary_report


# ---------------------------------------------------------------------------
# Criterion 8: a crash mid-round 2 is finished on resume
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("variant", ["in_process", "new_model", "subprocess"])
def test_crash_mid_round2_reattach_finishes_round2(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    reference: Any,
    variant: str,
) -> None:
    if variant == "in_process":
        model = Model(edge_config(), data=frame())
        crash_round_2(monkeypatch, model)
        _assert_round_2_open(model._study)
        assert model._round_number == 1  # the failed tune changed nothing
        kwargs: dict[str, Any] = {}
    else:
        storage = url(tmp_path)
        if variant == "new_model":
            crash_round_2(monkeypatch, Model(edge_config(), data=frame()), storage, "s")
        else:
            child = run_child_and_kill(storage, "s")
            assert child.returncode == 1, child.stderr[-2000:]
        _assert_round_2_open(load(storage, "s"))
        model = Model(edge_config(), data=frame())
        kwargs = {"storage": storage, "study_name": "s"}
    # Only the reattach's records: the crashed round 2 did log its expansion.
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="lizyml"):
        result = model.tune(resume=True, **kwargs)
    study = model._study if variant == "in_process" else load(storage, "s")
    _assert_round_2_finished(result, study, reference)
    # The open round's analysis is the record's: nothing is expanded again.
    messages = [r.getMessage() for r in caplog.records]
    assert [m for m in messages if "tune.start" in m]  # INFO was captured
    assert not [m for m in messages if "tune.expand" in m]
    assert model._round_number == 2
    assert model._space == list(_round_2_record(study).effective)


def test_in_process_exception_after_one_trial_finishes_round2(
    monkeypatch: pytest.MonkeyPatch, reference: Any
) -> None:
    """An exception out of the round (no RUNNING trial left): same rule."""
    model = Model(edge_config(), data=frame())
    model.tune(n_trials=ROUND_1_TRIALS)
    real = model._run_tune_round

    def one_trial_then_raise(tuner: Any, study: Any, objective: Any, plan: Any) -> None:
        tuner.n_trials = 1
        real(tuner, study, objective, plan)
        raise RuntimeError("interrupted")

    monkeypatch.setattr(model, "_run_tune_round", one_trial_then_raise)
    with pytest.raises(RuntimeError, match="interrupted"):
        model.tune(resume=True, expand_boundary=True)
    monkeypatch.undo()
    study = model._study
    assert end_key(2) not in study.user_attrs
    assert [t.state.name for t in study.trials] == ["COMPLETE"] * 3
    assert model._round_number == 1
    seen = spy_tuner(monkeypatch)
    result = model.tune(resume=True)
    assert seen["init"] == [{"n_trials": ROUND_2_TRIALS - 1, "timeout": None}]
    assert len(study.trials) == TOTAL
    assert [(r.round, r.n_trials) for r in result.rounds] == [
        (r.round, r.n_trials) for r in reference.rounds
    ]
    assert result.boundary_report == _round_2_record(study).boundary_report


def test_crash_before_end_closes_without_trials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reference: Any
) -> None:
    from tests.test_tuning._resume_helpers import die_at_attr

    storage = url(tmp_path)
    model = Model(edge_config(), data=frame())
    model.tune(n_trials=ROUND_1_TRIALS, storage=storage, study_name="s")
    die_at_attr(monkeypatch, end_key(2))
    with pytest.raises(ProcessDeath):
        model.tune(resume=True, expand_boundary=True)
    monkeypatch.undo()
    stored = load(storage, "s")
    assert round_key(2) in stored.user_attrs
    assert end_key(2) not in stored.user_attrs
    assert [t.state.name for t in stored.trials] == ["COMPLETE"] * TOTAL

    seen = spy_tuner(monkeypatch)
    fresh = Model(edge_config(), data=frame())
    result = fresh.tune(resume=True, storage=storage, study_name="s")
    assert seen["run"] == []
    assert seen["init"] == [{"n_trials": 0, "timeout": None}]
    stored = load(storage, "s")
    assert len(stored.trials) == TOTAL
    assert decode_round_end(stored.user_attrs[end_key(2)]).status == "completed"
    assert [(r.round, r.n_trials) for r in result.rounds] == [
        (r.round, r.n_trials) for r in reference.rounds
    ]


# ---------------------------------------------------------------------------
# Criterion 8a: the only trial of round 1 left RUNNING
# ---------------------------------------------------------------------------


def test_single_running_trial_round1_fails_then_round2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage = url(tmp_path)
    cfg = edge_config(n_trials=1)
    first = Model(cfg, data=frame())
    die_in_objective(monkeypatch, first, at_call=1)
    with pytest.raises(ProcessDeath):
        first.tune(storage=storage, study_name="s")
    monkeypatch.undo()
    stored = load(storage, "s")
    assert [t.state.name for t in stored.trials] == ["RUNNING"]
    assert round_key(1) in stored.user_attrs
    assert end_key(1) not in stored.user_attrs

    # With `output_dir`, so the snapshot would see `_run_dir` move (DC5).
    model = Model(cfg, data=frame(), output_dir=tmp_path / "out")
    before = snapshot_model(model)
    with pytest.raises(LizyMLError) as caught:
        model.tune(resume=True, storage=storage, study_name="s")
    assert caught.value.code is ErrorCode.TUNING_FAILED
    assert snapshot_model(model) == before
    assert (model._run_dir, model._provider) == (None, None)
    stored = load(storage, "s")
    assert [t.state.name for t in stored.trials] == ["FAIL"]
    end = decode_round_end(stored.user_attrs[end_key(1)])
    assert (end.status, end.best_after) == ("failed", None)

    # State F: a new round, with nothing to expand even when asked to.
    result = model.tune(
        resume=True, storage=storage, study_name="s", expand_boundary=True
    )
    assert [r.round for r in result.rounds] == [2]
    assert result.rounds[0].best_score_before is None
    assert result.rounds[0].n_trials == 1
    assert (result.trials[0].round, result.trials[0].state) == (1, "fail")
    assert [t.round for t in result.trials] == [1, 2]
    assert result.boundary_report is None
    lineage = classify_study(load(storage, "s").user_attrs, load(storage, "s").trials)
    assert lineage.rounds[1].effective == lineage.rounds[0].effective
    assert model._round_number == 2


# ---------------------------------------------------------------------------
# Criterion 8b (without `space=`): U2 on an open round with a RUNNING trial
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_trials": ROUND_2_TRIALS},
        {"expand_boundary": True},
        {},
        {"n_trials": None, "expand_boundary": None},
    ],
    ids=["n_trials", "expand_boundary", "omitted", "none"],
)
def test_u2_same_args_pass(storage: str, reference: Any, kwargs: Any) -> None:
    model = Model(edge_config(), data=frame())
    result = model.tune(resume=True, storage=storage, study_name="s", **kwargs)
    _assert_round_2_finished(result, load(storage, "s"), reference)


@pytest.mark.parametrize(
    ("kwargs", "surface"),
    [
        ({"n_trials": ROUND_2_TRIALS + 1}, "tune(n_trials=)"),
        ({"expand_boundary": False}, "tune(expand_boundary=)"),
    ],
    ids=["n_trials", "expand_boundary"],
)
def test_u2_different_args_refused(storage: str, kwargs: Any, surface: str) -> None:
    model = Model(edge_config(), data=frame())
    before = (snapshot_study(storage, "s"), snapshot_model(model))
    with pytest.raises(LizyMLError) as caught:
        model.tune(resume=True, storage=storage, study_name="s", **kwargs)
    assert caught.value.code is ErrorCode.CONFIG_INVALID
    assert caught.value.context["surface"] == surface
    assert caught.value.context["round"] == 2
    assert (snapshot_study(storage, "s"), snapshot_model(model)) == before
    _assert_round_2_open(load(storage, "s"))


def test_config_n_trials_warns(
    storage: str, monkeypatch: pytest.MonkeyPatch, reference: Any
) -> None:
    seen = spy_tuner(monkeypatch)
    model = Model(edge_config(n_trials=ROUND_2_TRIALS + 4), data=frame())
    with pytest.warns(UserWarning, match="n_trials"):
        result = model.tune(resume=True, storage=storage, study_name="s")
    # The recorded budget: 3 asked, #2 COMPLETE and #3 FAILed leave one.
    assert seen["init"] == [{"n_trials": 1, "timeout": None}]
    _assert_round_2_finished(result, load(storage, "s"), reference)


def test_after_closed_round_n_trials_free(storage: str) -> None:
    model = Model(edge_config(), data=frame())
    model.tune(resume=True, storage=storage, study_name="s")
    result = model.tune(resume=True, n_trials=ROUND_2_TRIALS + 1)
    assert [(r.round, r.n_trials) for r in result.rounds][-1] == (3, 4)
    assert _record(load(storage, "s"), 3).n_trials == ROUND_2_TRIALS + 1


def _record(study: Any, k: int) -> Any:
    lineage = classify_study(study.user_attrs, study.trials)
    return next(r for r in lineage.rounds if r.round == k)


# ---------------------------------------------------------------------------
# Criterion 16: the remaining timeout
# ---------------------------------------------------------------------------

TIMEOUT = 3600.0


def _timed_config(n_trials: int = ROUND_2_TRIALS) -> dict[str, Any]:
    cfg = edge_config(n_trials=n_trials)
    cfg["tuning"]["optuna"]["params"]["timeout"] = TIMEOUT
    return cfg


def _timed_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, n_trials: int = ROUND_2_TRIALS
) -> str:
    storage = url(tmp_path)
    model = Model(_timed_config(n_trials), data=frame())
    crash_round_2(monkeypatch, model, storage, "s")
    stored = load(storage, "s")
    assert _record(stored, 2).timeout == TIMEOUT
    assert stored.trials[CRASHED_TRIAL].state.name == "RUNNING"
    return storage


def _span(trial: Any) -> float:
    return float((trial.datetime_complete - trial.datetime_start).total_seconds())


def test_timeout_remaining_passed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage = _timed_crash(tmp_path, monkeypatch)
    round_2 = [t for t in load(storage, "s").trials if t.number >= ROUND_1_TRIALS]
    spent = sum(_span(t) for t in round_2 if t.state.name == "COMPLETE")
    assert spent > 0
    seen = spy_tuner(monkeypatch)
    Model(_timed_config(), data=frame()).tune(
        resume=True, storage=storage, study_name="s"
    )
    assert len(seen["init"]) == 1
    assert seen["init"][0]["n_trials"] == 1
    assert seen["init"][0]["timeout"] == pytest.approx(TIMEOUT - spent, abs=1e-6)
    assert len(seen["run"]) == 1


def test_timeout_exhausted_closes_without_trials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage = _timed_crash(tmp_path, monkeypatch)
    stored = load(storage, "s")
    record = copy.deepcopy(stored.user_attrs[round_key(2)])
    record["budget"]["timeout"] = ["float", 1e-6]
    stored.set_user_attr(round_key(2), record)
    seen = spy_tuner(monkeypatch)
    model = Model(_timed_config(), data=frame())
    result = model.tune(resume=True, storage=storage, study_name="s")
    assert seen["run"] == []
    assert seen["init"][0]["timeout"] <= 0
    stored = load(storage, "s")
    assert len(stored.trials) == ROUND_1_TRIALS + 2
    assert stored.trials[CRASHED_TRIAL].state.name == "FAIL"
    assert decode_round_end(stored.user_attrs[end_key(2)]).status == "completed"
    assert [(r.round, r.n_trials) for r in result.rounds] == [(1, 2), (2, 2)]


def test_second_crash_excludes_failed_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    budget = ROUND_2_TRIALS + 2
    storage = _timed_crash(tmp_path, monkeypatch, n_trials=budget)
    # The first reattach FAILs #3, then dies in its first trial (#4).
    first = Model(_timed_config(budget), data=frame())
    die_in_objective(monkeypatch, first, at_call=1)
    with pytest.raises(ProcessDeath):
        first.tune(resume=True, storage=storage, study_name="s")
    monkeypatch.undo()
    stored = load(storage, "s")
    states = [t.state.name for t in stored.trials]
    assert states == ["COMPLETE", "COMPLETE", "COMPLETE", "FAIL", "RUNNING"]
    # The process was down for a long time before the first reattach: the
    # trial it FAILed spans that downtime.
    sqlite_exec(
        storage,
        "UPDATE trials SET datetime_start = datetime(datetime_start, '-1000 seconds') "
        "WHERE number = ?",
        (CRASHED_TRIAL,),
    )
    stored = load(storage, "s")
    assert _span(stored.trials[CRASHED_TRIAL]) >= 1000
    round_2 = [t for t in stored.trials if t.number >= ROUND_1_TRIALS]
    spent = sum(_span(t) for t in round_2 if t.state.name == "COMPLETE")

    seen = spy_tuner(monkeypatch)
    result = Model(_timed_config(budget), data=frame()).tune(
        resume=True, storage=storage, study_name="s"
    )
    timeout = seen["init"][0]["timeout"]
    assert timeout == pytest.approx(TIMEOUT - spent, abs=1e-6)
    assert timeout > TIMEOUT - 1000  # the FAIL trial's span was not subtracted
    assert seen["init"][0]["n_trials"] == budget - 3
    assert result.rounds[-1].n_trials == budget
    assert load(storage, "s").trials[CRASHED_TRIAL + 1].state.name == "FAIL"


# ---------------------------------------------------------------------------
# Criteria 18 and 31: progress and the round summary
# ---------------------------------------------------------------------------


def _progress_of(model: Model, **kwargs: Any) -> list[tuple[int, int, int, int]]:
    seen: list[TuneProgressInfo] = []
    model.tune(resume=True, progress_callback=seen.append, **kwargs)
    return [
        (i.current_trial, i.total_trials, i.cumulative_trials, i.round) for i in seen
    ]


def test_progress_continues_k_plus_1(
    storage: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """K = 2 (#2 COMPLETE, #3 FAILed): the resumed trial is the round's 3rd."""
    stored = Model(edge_config(), data=frame())
    with_storage = _progress_of(stored, storage=storage, study_name="s")
    memory = Model(edge_config(), data=frame())
    crash_round_2(monkeypatch, memory)
    _assert_round_2_open(memory._study)
    in_memory = _progress_of(memory)
    expected = [(3, ROUND_2_TRIALS, TOTAL, 2)]
    assert with_storage == expected
    assert in_memory == with_storage


def test_roundsummary_counts_actual_trials_on_timeout() -> None:
    cfg = fast_config(n_trials=1000)
    cfg["tuning"]["optuna"]["params"]["timeout"] = 0.5
    model = Model(cfg, data=frame())
    result = model.tune()
    n = len(result.trials)
    assert 0 < n < 1000
    assert [(r.round, r.n_trials) for r in result.rounds] == [(1, n)]
    assert _record(model._study, 1).n_trials == 1000
