"""H-0150 part B slice 2: the budget's pre-write admission and resume.

Acceptance 7 (the device refusal before any write, and no refusal for a round
that runs no trial), acceptance 9 (a changed budget on resume, nothing
recorded), and the BLUEPRINT section 14.4 and ``estimate_memory`` claims the
part A review left behind.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import optuna
import pytest

from lizyml import Model
from lizyml.core.exceptions import ErrorCode, LizyMLError
from tests.test_core.test_h0150_estimate_tune_memory import (
    DEVICE_OFF_CPU,
    _cfg,
    _crashing_close,
    _df,
    _dump,
    _fixed,
    _Recorder,
    _space,
    _url,
    train_stub,  # noqa: F401 -- a fixture, used by name
)

BUDGET = 10**15


def _no_study_written(url: str) -> None:
    assert optuna.get_all_study_summaries(storage=url) == []


def _refused_for_device(cfg: dict[str, Any], tmp_path: Path, **kw: Any) -> None:
    url = _url(tmp_path)
    model = Model(cfg)
    with pytest.raises(LizyMLError) as err:
        model.tune(
            data=_df("regression"),
            storage=url,
            study_name="s",
            memory_budget_bytes=BUDGET,
            **kw,
        )
    assert err.value.code is ErrorCode.CONFIG_INVALID
    assert err.value.context["reason"] == "memory_budget_device_not_cpu"
    _no_study_written(url)  # refused before the round record is written
    assert model._study is None


# --- acceptance 7: a reachable non-CPU device is refused before any write --------


@pytest.mark.parametrize(
    ("label", "params", "dims"), DEVICE_OFF_CPU, ids=[c[0] for c in DEVICE_OFF_CPU]
)
def test_a_reachable_non_cpu_device_is_refused_before_any_write(
    label: str,
    params: dict[str, Any],
    dims: dict[str, Any],
    tmp_path: Path,
    train_stub: _Recorder,  # noqa: F811
) -> None:
    _refused_for_device(_cfg(space=_space(**dims), **params), tmp_path)
    assert train_stub.calls == []


def test_a_fixed_gpu_is_refused_even_when_a_candidate_cannot_be_assembled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    train_stub: _Recorder,  # noqa: F811
) -> None:
    _fixed(monkeypatch, device_type="gpu")
    space = _space(
        validation_ratio={
            "type": "float",
            "low": 0.0,
            "high": 0.3,
            "category": "training",
        }
    )
    _refused_for_device(_cfg(space=space, mode="merge"), tmp_path)
    assert train_stub.calls == []


def test_without_a_budget_the_device_is_not_judged(
    tmp_path: Path,
    train_stub: _Recorder,  # noqa: F811
) -> None:
    cfg = _cfg(space=_space(), n_trials=2, device_type="gpu")
    Model(cfg).tune(data=_df("regression"))  # as before: no refusal
    assert len(train_stub.calls) > 0


def test_a_round_with_no_trial_closes_as_before_despite_budget_and_gpu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    train_stub: _Recorder,  # noqa: F811
) -> None:
    cfg = _cfg(space=_space(), n_trials=2, device_type="gpu")
    df = _df("regression")
    url = _url(tmp_path)
    with monkeypatch.context() as m:
        _crashing_close(m)
        with pytest.raises(RuntimeError):
            Model(cfg).tune(data=df, storage=url, study_name="s")
    result = Model(cfg).tune(
        data=df, resume=True, storage=url, study_name="s", memory_budget_bytes=1
    )
    assert result.best_params  # the open round closed; the budget was not used
    # The value check always runs, even for a round that runs no trial.
    with pytest.raises(LizyMLError) as err:
        Model(cfg).tune(
            data=df, resume=True, storage=url, study_name="s", memory_budget_bytes=0
        )
    assert err.value.code is ErrorCode.CONFIG_INVALID


# --- acceptance 9: resume with another budget ----------------------------------------


@pytest.mark.parametrize("resume_budget", [BUDGET // 7, None])
def test_a_changed_budget_resumes_an_open_round_and_records_nothing(
    resume_budget: int | None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    train_stub: _Recorder,  # noqa: F811
) -> None:
    cfg = _cfg(space=_space(), auto=False, n_trials=2)
    df = _df("regression")
    url = _url(tmp_path)
    with monkeypatch.context() as m:
        m.setattr(
            "lizyml.core._model_tuning.ModelTuningMixin._run_tune_round",
            lambda self, *a, **k: (_ for _ in ()).throw(RuntimeError("died")),
        )
        with pytest.raises(RuntimeError):
            Model(cfg).tune(
                data=df, storage=url, study_name="s", memory_budget_bytes=BUDGET
            )
    result = Model(cfg).tune(
        data=df,
        resume=True,
        storage=url,
        study_name="s",
        memory_budget_bytes=resume_budget,
    )
    assert result.best_params and len(train_stub.calls) > 0
    dump = repr(_dump(tmp_path / "study.db")).lower()
    assert "memory_budget" not in dump and "memory" not in dump
    assert str(BUDGET) not in dump and str(BUDGET // 7) not in dump


@pytest.mark.parametrize("resume_budget", [BUDGET, None])
def test_a_round_whose_trials_were_all_refused_earlier_reports_zero_now(
    resume_budget: int | None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    train_stub: _Recorder,  # noqa: F811
) -> None:
    cfg = _cfg(space=_space(), auto=False, n_trials=2)
    df = _df("regression")
    url = _url(tmp_path)
    with monkeypatch.context() as m:
        _crashing_close(m)
        with pytest.raises(RuntimeError):
            Model(cfg).tune(data=df, storage=url, study_name="s", memory_budget_bytes=1)
    study = optuna.load_study(study_name="s", storage=url)
    assert {t.state.name for t in study.trials} == {"FAIL"}
    with pytest.raises(LizyMLError) as err:
        Model(cfg).tune(
            data=df,
            resume=True,
            storage=url,
            study_name="s",
            memory_budget_bytes=resume_budget,
        )
    assert err.value.code is ErrorCode.TUNING_FAILED
    assert err.value.context["memory_refused_this_call"] == 0
    assert err.value.context["memory_budget_bytes"] == resume_budget
    assert train_stub.calls == []


# --- the part A review minor findings, pinned ---


def test_a_scalar_string_max_bin_by_feature_choice_is_admitted_but_unresolved() -> None:
    space = _space(
        max_bin_by_feature={"type": "categorical", "choices": ["16,16", "8,8"]}
    )
    est = Model(_cfg(space=space, auto=False)).estimate_tune_memory(_df("regression"))
    assert est.status == "incomplete"
    assert "max_bin_by_feature" in est.unresolved


def test_a_list_valued_max_bin_by_feature_choice_is_refused() -> None:
    space = _space(
        max_bin_by_feature={"type": "categorical", "choices": [[16, 16], [8, 8]]}
    )
    with pytest.raises(LizyMLError) as err:
        Model(_cfg(space=space, auto=False)).estimate_tune_memory(_df("regression"))
    assert err.value.code is ErrorCode.CONFIG_INVALID


def test_the_estimate_does_not_mirror_fit_only_splitter_refusals() -> None:
    cfg = _cfg(space=_space())
    cfg["split"] = {"method": "stratified_kfold", "n_splits": 2}
    df = _df("regression")
    assert Model(cfg).estimate_memory(df).estimate_bytes > 0
    with pytest.raises(LizyMLError):
        Model(cfg).fit(df)
