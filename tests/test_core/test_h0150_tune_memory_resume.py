"""H-0150 part A: ``estimate_tune_memory`` on a stored study (acceptance 4, 5).

Built on the H-0122 crash scenario (``_resume_helpers.crash_round_2``): round
1 ran 2 trials over ``num_leaves`` [4, 5]; round 2 expanded that range, asked
for 3 trials, and died in its 2nd one, which stays RUNNING. So the open
round's recorded range differs from the config's, and the study holds a
RUNNING row the next tune would fail.

Also the warning attribution of the planning that moved out of ``tune``
(Codex implementation review round 1, finding 6): a warning that named the
caller before the move still names it, from ``tune`` and from the estimate.
"""

from __future__ import annotations

import shutil
import sqlite3
import warnings
from pathlib import Path
from typing import Any

import optuna
import pytest

from lizyml import Model
from lizyml.core.exceptions import LizyMLError
from lizyml.core.types import TuneMemoryEstimate
from lizyml.tuning.lineage import classify_study
from tests.test_tuning._resume_helpers import (
    CRASHED_TRIAL,
    ROUND_2_TRIALS,
    crash_round_2,
    edge_config,
    fast_config,
    frame,
    load,
    url,
)


@pytest.fixture(scope="module")
def open_template(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("h0150_open")
    with pytest.MonkeyPatch.context() as mp:
        crash_round_2(mp, Model(edge_config(), data=frame()), url(root), "s")
    return root / "s.db"


@pytest.fixture
def storage(open_template: Path, tmp_path: Path) -> str:
    shutil.copy(open_template, tmp_path / "s.db")
    return url(tmp_path)


def _db(storage: str) -> Path:
    return Path(storage.removeprefix("sqlite:///"))


def _dump(path: Path) -> dict[str, list[Any]]:
    con = sqlite3.connect(path)
    try:
        tables = [
            r[0]
            for r in con.execute(
                "select name from sqlite_master where type='table' order by name"
            )
        ]
        return {t: con.execute(f"select * from {t}").fetchall() for t in tables}  # noqa: S608
    finally:
        con.close()


def _estimate(storage: str, **kw: Any) -> TuneMemoryEstimate:
    return Model(edge_config(), data=frame()).estimate_tune_memory(
        resume=True, storage=storage, study_name="s", **kw
    )


def _recorded_num_leaves(storage: str) -> Any:
    study = load(storage, "s")
    lineage = classify_study(study.user_attrs, study.trials)
    assert lineage.state == "open" and lineage.last_round is not None
    assert lineage.last_round.round == 2
    return {d.name: d for d in lineage.last_round.effective}["num_leaves"]


# --- acceptance 4: the open round's recorded range ----------------------------------


def test_the_open_round_uses_its_recorded_range_not_the_configs(storage: str) -> None:
    recorded = _recorded_num_leaves(storage)
    configured = edge_config()["tuning"]["optuna"]["space"]["num_leaves"]
    assert recorded.high != configured["high"]  # reading the config would fail
    est = _estimate(storage)
    assert est.round == 2 and est.status == "complete"
    assert est.envelope is not None and est.envelope.num_leaves == recorded.high


def test_an_out_of_range_waiting_trial_is_incomplete_and_not_a_candidate(
    storage: str,
) -> None:
    before = _estimate(storage)
    assert before.envelope is not None
    recorded = _recorded_num_leaves(storage)
    load(storage, "s").enqueue_trial(
        {"num_leaves": recorded.high * 64, "learning_rate": 0.1},
        skip_if_exists=False,
    )
    after = _estimate(storage)
    assert after.status == "incomplete" and after.waiting_trials == 1
    assert after.envelope == before.envelope  # the WAITING values are not read
    assert after.n_candidates == before.n_candidates


# --- acceptance 5: nothing is written, the RUNNING row included ---------------------


def test_a_running_trial_and_the_tables_are_unchanged(storage: str) -> None:
    states = {t.number: t.state for t in load(storage, "s").trials}
    assert states[CRASHED_TRIAL] is optuna.trial.TrialState.RUNNING
    before = _dump(_db(storage))
    est = _estimate(storage)
    assert est.n_candidates > 0
    assert _dump(_db(storage)) == before
    after = {t.number: t.state for t in load(storage, "s").trials}
    assert after == states


def test_a_changed_open_round_n_trials_is_the_same_refusal(storage: str) -> None:
    kwargs = {"n_trials": ROUND_2_TRIALS + 1}
    with pytest.raises(LizyMLError) as est_err:
        _estimate(storage, **kwargs)
    with pytest.raises(LizyMLError) as tune_err:
        Model(edge_config(), data=frame()).tune(
            resume=True, storage=storage, study_name="s", **kwargs
        )
    assert est_err.value.code is tune_err.value.code
    assert est_err.value.context == tune_err.value.context
    assert est_err.value.context["surface"] == "tune(n_trials=)"


# --- finding 6: warnings keep their attribution ------------------------------------


def _warned(
    match: str, call: Any, *, at_call: bool = False
) -> list[warnings.WarningMessage]:
    """The warnings matching *match*. With *at_call*, each must name this
    file and the line of the single-line lambda *call*, not only the file."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        call()
    hits = [w for w in caught if match in str(w.message)]
    assert hits, f"no warning matching {match!r}"
    if at_call:
        line = call.__code__.co_firstlineno
        assert [(Path(w.filename).name, w.lineno) for w in hits] == [
            (Path(__file__).name, line)
        ]
    return hits


def _from(hits: list[warnings.WarningMessage]) -> list[str]:
    return [Path(w.filename).name for w in hits]


def test_the_open_round_n_trials_warning_names_the_caller(storage: str) -> None:
    def model() -> Model:
        return Model(edge_config(n_trials=ROUND_2_TRIALS + 4), data=frame())

    kw: dict[str, Any] = {"resume": True, "storage": storage, "study_name": "s"}
    est = _warned(
        "is ignored", lambda: model().estimate_tune_memory(**kw), at_call=True
    )
    assert _from(est) == [Path(__file__).name]
    tuned = _warned("is ignored", lambda: model().tune(**kw), at_call=True)
    assert _from(tuned) == [Path(__file__).name]


def _drift(monkeypatch: pytest.MonkeyPatch) -> None:
    from lizyml.core import _tuning_resume
    from lizyml.core._tuning_identity import VersionDifference

    monkeypatch.setattr(
        _tuning_resume,
        "version_differences",
        lambda stored, current: (VersionDifference("lightgbm", "0.0", "9.9"),),
    )


def test_the_version_drift_warning_names_the_caller(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage = url(tmp_path)
    df = frame()
    Model(fast_config(), data=df).tune(storage=storage, study_name="s")
    _drift(monkeypatch)
    kw: dict[str, Any] = {"resume": True, "storage": storage, "study_name": "s"}
    msg = "other library versions"
    est = _warned(
        msg,
        lambda: Model(fast_config(), data=df).estimate_tune_memory(**kw),
        at_call=True,
    )
    assert _from(est) == [Path(__file__).name]
    tuned = _warned(msg, lambda: Model(fast_config(), data=df).tune(**kw), at_call=True)
    assert _from(tuned) == [Path(__file__).name]
    # restore_tuning reaches the same check one frame shallower; it keeps
    # naming the tuning module, as it did before the planning moved.
    restored = _warned(
        msg,
        lambda: Model(fast_config(), data=df).restore_tuning(
            storage=storage, study_name="s"
        ),
    )
    assert _from(restored) == ["_model_tuning.py"]


def test_the_classification_kfold_warning_keeps_its_module() -> None:
    cfg = fast_config("binary", n_trials=1)
    assert cfg["split"]["method"] == "kfold"
    tuned = _warned("kfold", lambda: Model(cfg, data=frame("binary")).tune())
    assert _from(tuned) == ["_model_tuning.py"]
    est = _warned(
        "kfold", lambda: Model(cfg, data=frame("binary")).estimate_tune_memory()
    )
    assert _from(est) == ["_model_tuning.py"]
