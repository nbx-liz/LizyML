"""H-0122 criterion 36: ``Model.restore_tuning`` (§7.7).

A tune whose round closed (``:end`` written) but whose caller never received
the result leaves the study in state C. ``restore_tuning`` returns that
result without adding a trial: these tests compare it with the result an
uninterrupted tune returned, check the Model is committed as a tune would
leave it (so ``fit`` and a later ``tune(resume=True)`` behave the same), and
check that it never writes to the study.

The refusals with their study and Model snapshots are rows of
``test_resume_refusal_invariance.py`` (criterion 22); here are their order
(design R-6) and the argument checks.
"""

from __future__ import annotations

import copy
import math
import shutil
from pathlib import Path
from typing import Any

import optuna
import pandas as pd
import pytest

from lizyml import Model
from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.core.types.tuning_result import TuningResult
from lizyml.tuning import read_study_lineage
from lizyml.tuning import tuner as tuner_module
from lizyml.tuning.lineage import STUDY_KEY, end_key, round_key
from lizyml.tuning.search_space import detect_boundary
from tests.test_tuning._resume_helpers import (
    ProcessDeath,
    crash_round_2,
    die_after_attr,
    edge_config,
    failed_round_1,
    fast_config,
    frame,
    load,
    round_record,
    set_attr,
    snapshot_model,
    snapshot_study,
    spy_fit_params,
    url,
)

CI = ErrorCode.CONFIG_INVALID
TF = ErrorCode.TUNING_FAILED


def _trials(result: TuningResult) -> list[tuple[Any, ...]]:
    """Trials as comparable tuples; a FAIL trial's NaN score compares as None."""
    return [
        (
            t.number,
            t.round,
            t.state,
            t.params,
            None if math.isnan(t.score) else t.score,
        )
        for t in result.trials
    ]


def assert_same_result(got: TuningResult, want: TuningResult) -> None:
    """Field by field (a FAIL trial's NaN score defeats dataclass equality)."""
    assert got.best_model_params == want.best_model_params
    assert got.best_smart_params == want.best_smart_params
    assert got.best_training_params == want.best_training_params
    assert got.best_score == want.best_score
    assert got.metric_name == want.metric_name
    assert got.direction == want.direction
    assert got.rounds == want.rounds
    assert _trials(got) == _trials(want)
    assert got.boundary_report == want.boundary_report


@pytest.fixture(scope="module")
def live(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, TuningResult]:
    """Two rounds run to the end; the second round's result as returned."""
    root = tmp_path_factory.mktemp("live")
    storage = url(root)
    model = Model(edge_config(), data=frame())
    model.tune(storage=storage, study_name="s")
    result = model.tune(resume=True, expand_boundary=True)
    assert [r.round for r in result.rounds] == [1, 2]
    return root / "s.db", result


def _copy(live: tuple[Path, TuningResult], tmp_path: Path) -> str:
    shutil.copy(live[0], tmp_path / "s.db")
    return url(tmp_path)


# ---------------------------------------------------------------------------
# The restored result
# ---------------------------------------------------------------------------


def test_restore_equals_live_result(
    live: tuple[Path, TuningResult], tmp_path: Path
) -> None:
    storage = _copy(live, tmp_path)
    model = Model(edge_config(), data=frame())
    restored = model.restore_tuning(storage, "s")
    assert_same_result(restored, live[1])
    assert restored.boundary_report is not None
    assert restored.boundary_report.expanded_names


def test_restore_writes_nothing(
    live: tuple[Path, TuningResult], tmp_path: Path
) -> None:
    storage = _copy(live, tmp_path)
    before = snapshot_study(storage, "s")
    Model(edge_config(), data=frame()).restore_tuning(storage, "s")
    assert snapshot_study(storage, "s") == before
    assert round_key(3) not in load(storage, "s").user_attrs


#: Tune-owned fields compared by value between the live and the restored Model.
_COMMITTED = (
    "_round_number",
    "_rounds",
    "_space",
    "_used_default_space",
    "_tuning_fixed_params",
    "_tuning_storage",
    "_tuning_study_name",
)


@pytest.mark.parametrize("mode", ["replace", "merge"])
def test_restore_commits_the_tune_state(tmp_path: Path, mode: str) -> None:
    """The Model is left as the tune that closed round 2 left its own."""
    storage = url(tmp_path)
    cfg = fast_config(mode=mode)
    live_model = Model(cfg, data=frame())
    live_model.tune(storage=storage, study_name="s")
    live_model.tune(resume=True)

    model = Model(copy.deepcopy(cfg), data=frame())
    result = model.restore_tuning(storage, "s")
    assert model._tuning_result is result
    for name in _COMMITTED:
        assert getattr(model, name) == getattr(live_model, name), name
    assert model._round_number == 2
    assert model._space == list(round_record(storage, 2).effective)
    assert model._study is not None and model._study.study_name == "s"
    pd.testing.assert_frame_equal(model._X, live_model._X)
    pd.testing.assert_series_equal(model._y, live_model._y)
    # Transients of a tune call, not committed (design §4.2).
    assert model._provider is None
    assert model._run_dir is None


def test_reporting_surfaces_match_live(
    live: tuple[Path, TuningResult], tmp_path: Path
) -> None:
    """``tuning_table`` / ``boundary_table`` / ``tuning_plot`` read the
    restored result exactly as they read the live one."""
    storage = _copy(live, tmp_path)
    model = Model(edge_config(), data=frame())
    model.restore_tuning(storage, "s")
    holder = Model(edge_config(), data=frame())
    holder._tuning_result = live[1]
    pd.testing.assert_frame_equal(model.tuning_table(), holder.tuning_table())
    pd.testing.assert_frame_equal(model.boundary_table(), holder.boundary_table())
    assert model.tuning_plot().to_dict() == holder.tuning_plot().to_dict()


def test_crash_after_end_then_restore_then_round3(
    live: tuple[Path, TuningResult],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Round 2 closes, then the process dies before the caller gets it."""
    storage = url(tmp_path)
    first = Model(edge_config(), data=frame())
    first.tune(storage=storage, study_name="s")
    die_after_attr(monkeypatch, end_key(2))
    with pytest.raises(ProcessDeath):
        first.tune(resume=True, expand_boundary=True)
    monkeypatch.undo()
    assert first._round_number == 1  # the failed call committed nothing
    assert read_study_lineage(storage, "s").state == "completed"

    # `data` given to the call, by tune's priority (the constructor has none).
    model = Model(edge_config())
    before = snapshot_study(storage, "s")
    restored = model.restore_tuning(storage, "s", data=frame())
    assert snapshot_study(storage, "s") == before
    assert_same_result(restored, live[1])

    result = model.tune(frame(), resume=True)
    assert [r.round for r in result.rounds] == [1, 2, 3]
    assert model._round_number == 3
    assert round_key(3) in load(storage, "s").user_attrs
    assert {t.round for t in result.trials} == {1, 2, 3}


def test_fit_after_restore_uses_best_and_fixed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """INV-R2: ``fit`` after a restore trains as ``fit`` after the live tune.

    Merge mode, so the study's ``fixed`` (``first_metric_only``) is non-empty:
    a restore that did not carry it would fit without it.
    """
    storage = url(tmp_path)
    cfg = fast_config(mode="merge")
    live_model = Model(cfg, data=frame())
    live_model.tune(storage=storage, study_name="s")
    live_result = live_model.tune(resume=True)
    live_fit = spy_fit_params(monkeypatch, live_model)
    live_model.fit()
    monkeypatch.undo()

    model = Model(copy.deepcopy(cfg), data=frame())
    restored = model.restore_tuning(storage, "s")
    assert_same_result(restored, live_result)
    assert (
        model._tuning_fixed_params == load(storage, "s").user_attrs[STUDY_KEY]["fixed"]
    )
    seen = spy_fit_params(monkeypatch, model)
    model.fit()
    got, want = seen[-1], live_fit[-1]
    for key in ("model_params", "smart_params", "training_overrides"):
        assert got[key] == want[key], key
    assert got["model_params"]["first_metric_only"] is True
    for name, value in restored.best_model_params.items():
        assert got["model_params"][name] == value


def test_restore_on_fitted_model_keeps_fit(
    live: tuple[Path, TuningResult], tmp_path: Path
) -> None:
    """H-0094 decision 13: only the tuning result is replaced."""
    storage = _copy(live, tmp_path)
    model = Model(edge_config(), data=frame())
    model.fit()
    fit_result, refit_result = model._fit_result, model._refit_result
    params = model.params_table()
    metrics = model.evaluate()
    assert model._tuning_result is None

    restored = model.restore_tuning(storage, "s")
    assert model._fit_result is fit_result
    assert model._refit_result is refit_result
    pd.testing.assert_frame_equal(model.params_table(), params)
    assert model.evaluate() == metrics
    assert model._tuning_result is restored


def test_version_difference_warns(
    live: tuple[Path, TuningResult], tmp_path: Path
) -> None:
    storage = _copy(live, tmp_path)
    record = copy.deepcopy(load(storage, "s").user_attrs[STUDY_KEY])
    record["versions"]["lightgbm"] = "0.0.1"
    set_attr(storage, "s", STUDY_KEY, record)
    before = snapshot_study(storage, "s")
    with pytest.warns(UserWarning, match="other library versions"):
        restored = Model(edge_config(), data=frame()).restore_tuning(storage, "s")
    assert_same_result(restored, live[1])
    assert snapshot_study(storage, "s") == before


def _threshold_config() -> dict[str, Any]:
    cfg = fast_config(n_trials=4)
    # Seeded so round 1's best `lambda_l1` is ~0.2: the lower edge (asserted).
    cfg["training"]["seed"] = 4
    cfg["tuning"]["optuna"]["space"]["lambda_l1"] = {
        "type": "float",
        "low": 0.0,
        "high": 1.0,
    }
    return cfg


def test_threshold_report_reused(tmp_path: Path) -> None:
    """A non-default ``boundary_threshold``: the restored report is the
    record's, not one recomputed at the default threshold.

    Round 1's best ``lambda_l1`` sits at position ``p`` of [0, 1] with
    ``0.05 <= min(p, 1 - p) < 0.4`` (asserted, seeded): only a threshold above
    ``min(p, 1 - p)`` flags the edge. At the lower edge the 0.0 floor leaves
    the range unchanged, so the report differs from a clamp-free one only in
    its judgement (spec criterion 36's [0, 1] / 0.08 / 0.10 case).
    """
    storage = url(tmp_path)
    cfg = _threshold_config()
    model = Model(cfg, data=frame())
    first = model.tune(storage=storage, study_name="s")
    p = float(first.best_params["lambda_l1"])
    margin = min(p, 1.0 - p)
    assert 0.05 <= margin < 0.4, p
    threshold = round(margin + 0.01, 6)
    live = model.tune(resume=True, expand_boundary=True, boundary_threshold=threshold)

    restored = Model(copy.deepcopy(cfg), data=frame()).restore_tuning(storage, "s")
    assert_same_result(restored, live)
    report = restored.boundary_report
    assert report is not None
    row = {d.name: d for d in report.dims}["lambda_l1"]
    assert p < 0.5  # the seeded precondition: the lower edge
    assert row.edge == "lower"
    # The 0.0 floor holds the range: judged `lower`, nothing expanded.
    assert (row.expanded, row.new_low, row.new_high) == (False, None, None)
    lambda_l1 = {d.name: d for d in round_record(storage, 2).effective}["lambda_l1"]
    assert (lambda_l1.low, lambda_l1.high) == (0.0, 1.0)
    previous = list(round_record(storage, 1).effective)
    assert report != detect_boundary(previous, dict(first.best_params), 0.05)
    default_row = {
        d.name: d for d in detect_boundary(previous, dict(first.best_params), 0.05).dims
    }["lambda_l1"]
    assert default_row.edge == "none"


# ---------------------------------------------------------------------------
# Arguments and refusal order (design R-5, R-6)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("args", "surface"),
    [
        (("<url>", None), "tune(storage=)"),
        ((None, "s"), "tune(study_name=)"),
        ((None, None), "restore_tuning(storage=, study_name=)"),
    ],
    ids=["storage_only", "name_only", "neither"],
)
def test_arguments(tmp_path: Path, args: tuple[Any, Any], surface: str) -> None:
    model = Model(edge_config(), data=frame())
    model.tune()
    storage = url(tmp_path) if args[0] == "<url>" else None
    before = snapshot_model(model)
    with pytest.raises(LizyMLError) as caught:
        model.restore_tuning(storage, args[1])  # type: ignore[arg-type]
    assert caught.value.code is CI
    assert caught.value.context["surface"] == surface
    assert snapshot_model(model) == before
    assert not (tmp_path / "s.db").exists()


def test_no_tuning_config(tmp_path: Path) -> None:
    cfg = fast_config()
    del cfg["tuning"]
    with pytest.raises(LizyMLError) as caught:
        Model(cfg, data=frame()).restore_tuning(url(tmp_path), "s")
    assert caught.value.code is CI
    assert not (tmp_path / "s.db").exists()


def test_optuna_is_required_before_data_and_storage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The extras registry row (H-0119 table, design D-e): without optuna the
    call raises ``OPTIONAL_DEP_MISSING`` before it loads data (this Model has
    none) or contacts the storage."""

    def missing() -> Any:
        raise LizyMLError(
            code=ErrorCode.OPTIONAL_DEP_MISSING,
            user_message="optuna missing (test)",
            context={"package": "optuna"},
        )

    monkeypatch.setattr(tuner_module, "require_optuna", missing)
    with pytest.raises(LizyMLError) as caught:
        Model(edge_config()).restore_tuning(url(tmp_path), "s")
    assert caught.value.code is ErrorCode.OPTIONAL_DEP_MISSING
    assert not (tmp_path / "s.db").exists()


def test_absent_is_tuning_failed_and_creates_no_study(tmp_path: Path) -> None:
    storage = url(tmp_path)
    with pytest.raises(LizyMLError) as caught:
        Model(edge_config(), data=frame()).restore_tuning(storage, "s")
    assert caught.value.code is TF
    assert caught.value.context["state"] == "absent"
    assert "s" not in optuna.get_all_study_names(storage)


def test_open_is_refused_and_running_stays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage = url(tmp_path)
    crash_round_2(monkeypatch, Model(edge_config(), data=frame()), storage, "s")
    with pytest.raises(LizyMLError) as caught:
        Model(edge_config(), data=frame()).restore_tuning(storage, "s")
    assert caught.value.code is CI
    assert caught.value.context["state"] == "open"
    assert caught.value.context["round"] == 2
    assert "tune(resume=True)" in caught.value.user_message
    assert load(storage, "s").trials[-1].state.name == "RUNNING"


def test_nothing_to_restore_wins_over_a_mismatch(tmp_path: Path) -> None:
    """R-6: a ``failed`` study with changed data is ``TUNING_FAILED``."""
    storage = url(tmp_path)
    failed_round_1(storage)
    changed = frame()
    changed.loc[0, "feat_a"] = float(changed.loc[0, "feat_a"]) + 1.0
    with pytest.raises(LizyMLError) as caught:
        Model(edge_config(n_trials=1), data=changed).restore_tuning(storage, "s")
    assert caught.value.code is TF
    assert caught.value.context["state"] == "failed"


def test_mismatched_objective_on_c_is_refused(
    live: tuple[Path, TuningResult], tmp_path: Path
) -> None:
    storage = _copy(live, tmp_path)
    changed = frame()
    changed.loc[0, "feat_a"] = float(changed.loc[0, "feat_a"]) + 1.0
    with pytest.raises(LizyMLError) as caught:
        Model(edge_config(), data=changed).restore_tuning(storage, "s")
    assert caught.value.code is CI
    assert "data.values" in caught.value.context["mismatched"]
