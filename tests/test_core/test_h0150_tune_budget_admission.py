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
from lizyml.estimators.lgbm.provider import LGBMProvider
from lizyml.tuning.lineage import decode_round_end, end_key
from tests.test_core.test_h0150_estimate_tune_memory import (
    DEVICE_OFF_CPU,
    N_FOLDS,
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
from tests.test_core.test_h0150_tune_memory_candidates import (
    _ForeignAdapterProvider,
)

BUDGET = 10**15


def _no_study_written(url: str) -> None:
    assert optuna.get_all_study_summaries(storage=url) == []


def _end_record(url: str, k: int = 1) -> Any:
    """The decoded end record of round *k*, or ``None`` while it is open."""
    attrs = optuna.load_study(study_name="s", storage=url).user_attrs
    raw = attrs.get(end_key(k))
    return None if raw is None else decode_round_end(raw)


def _trial_tables(path: Path) -> dict[str, list[Any]]:
    return {t: rows for t, rows in _dump(path).items() if t.startswith("trial")}


def _no_budget_in(path: Path, *budgets: int | None) -> None:
    dump = repr(_dump(path)).lower()
    assert "memory_budget" not in dump and "memory" not in dump
    for value in budgets:
        if value is not None:
            assert str(value) not in dump


def _use(monkeypatch: pytest.MonkeyPatch, provider: Any) -> None:
    """Hand *provider* to every module ``tune`` resolves it in."""
    for module in ("_model_tuning", "_tune_planning"):
        monkeypatch.setattr(f"lizyml.core.{module}.get_provider", lambda _c: provider)


class _Incapable:
    """An LGBM provider without the memory capability (``isinstance`` is
    False once the methods are hidden)."""

    def __init__(self) -> None:
        self._p = LGBMProvider()

    def __getattr__(self, name: str) -> Any:
        if name in ("estimate_memory", "memory_role", "device_is_cpu"):
            raise AttributeError(name)
        return getattr(self._p, name)


class _PartAOnly(LGBMProvider):
    """A provider with exactly part A's ``MemoryEstimateProvider`` methods:
    ``finalized_device = None`` blocks the method for ``isinstance``."""

    finalized_device = None  # type: ignore[assignment]


def _exhausted_open_round(
    cfg: dict[str, Any], df: Any, url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A round whose trials all ran but whose end record was never written."""
    with monkeypatch.context() as m:
        _crashing_close(m)
        with pytest.raises(RuntimeError):
            Model(cfg).tune(data=df, storage=url, study_name="s")
    assert _end_record(url) is None


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
    _exhausted_open_round(cfg, df, url, monkeypatch)
    # The value check always runs, even for a round that runs no trial: on the
    # still-open round, before planning, and the round stays open.
    planned: list[int] = []
    real_plan = Model._plan_tune

    def spy(self: Any, *a: Any, **k: Any) -> Any:
        planned.append(1)
        return real_plan(self, *a, **k)

    before = _dump(tmp_path / "study.db")
    with monkeypatch.context() as m:
        m.setattr(Model, "_plan_tune", spy)
        with pytest.raises(LizyMLError) as err:
            Model(cfg).tune(
                data=df, resume=True, storage=url, study_name="s", memory_budget_bytes=0
            )
    assert err.value.code is ErrorCode.CONFIG_INVALID
    assert err.value.context["reason"] == "invalid_memory_budget"
    assert planned == []
    assert _end_record(url) is None
    assert _dump(tmp_path / "study.db") == before
    result = Model(cfg).tune(
        data=df, resume=True, storage=url, study_name="s", memory_budget_bytes=1
    )
    assert result.best_params  # the open round closed; the budget was not used
    assert _end_record(url).status == "completed"
    # After the close a new round plans trials, and 0 is refused there as well.
    with pytest.raises(LizyMLError) as err:
        Model(cfg).tune(
            data=df, resume=True, storage=url, study_name="s", memory_budget_bytes=0
        )
    assert err.value.code is ErrorCode.CONFIG_INVALID


@pytest.mark.parametrize(
    "provider",
    [_Incapable, _ForeignAdapterProvider, _PartAOnly],
    ids=["incapable", "foreign", "part-a-only"],
)
def test_a_round_with_no_trial_skips_the_capability_checks(
    provider: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    train_stub: _Recorder,  # noqa: F811
) -> None:
    cfg = _cfg(space=_space(), n_trials=2)
    df = _df("regression")
    url = _url(tmp_path)
    _exhausted_open_round(cfg, df, url, monkeypatch)
    _use(monkeypatch, provider())
    result = Model(cfg).tune(
        data=df, resume=True, storage=url, study_name="s", memory_budget_bytes=10**9
    )
    assert result.best_params  # closed as without a budget
    assert _end_record(url).status == "completed"


# --- policy 5: an adapter the provider cannot estimate, before any write ---------


def test_an_adapter_that_cannot_be_estimated_is_refused_before_any_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    train_stub: _Recorder,  # noqa: F811
) -> None:
    _use(monkeypatch, _ForeignAdapterProvider())
    url = _url(tmp_path)
    model = Model(_cfg(space=_space(), n_trials=2))
    with pytest.raises(LizyMLError) as err:
        model.tune(
            data=_df("regression"),
            storage=url,
            study_name="s",
            memory_budget_bytes=BUDGET,
        )
    assert err.value.code is ErrorCode.CONFIG_INVALID
    assert err.value.context["reason"] == "memory_estimate_unsupported"
    _no_study_written(url)
    assert model._study is None
    assert train_stub.calls == []


@pytest.mark.parametrize(
    "exc",
    [
        TypeError("the candidate point cannot be estimated"),
        # Also CONFIG_INVALID, so the probe must decide on the reason.
        LizyMLError(
            code=ErrorCode.CONFIG_INVALID,
            user_message="the candidate point has fewer than 1 round",
            context={"reason": "memory_estimate_unevaluable"},
        ),
    ],
    ids=["type-error", "unevaluable"],
)
def test_an_unevaluable_base_point_is_left_to_the_trial_guard(
    exc: Exception,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    train_stub: _Recorder,  # noqa: F811
) -> None:
    # Only memory_estimate_unsupported refuses at admission; another failure of
    # a candidate point estimate starts the round, and each trial is judged.
    real = LGBMProvider.estimate_memory
    seen = {"n": 0}

    def first_raises(self: LGBMProvider, **kw: Any) -> Any:
        seen["n"] += 1
        if seen["n"] == 1:
            raise exc
        return real(self, **kw)

    monkeypatch.setattr(LGBMProvider, "estimate_memory", first_raises)
    result = Model(_cfg(space=_space(), auto=False, n_trials=2)).tune(
        data=_df("regression"), memory_budget_bytes=BUDGET
    )
    assert result.best_params
    assert seen["n"] == 3  # one candidate point (learning_rate low), then one per trial
    assert len(train_stub.calls) == 2 * N_FOLDS


# --- round 2 finding 1: the probe evaluates the envelope's candidate points ----------


class _ForeignWhen(LGBMProvider):
    """A capable provider whose factory builds a foreign adapter only for the
    model parameters *when* accepts (``build_estimator_factory`` may depend on
    the parameters, ``lizyml/estimators/provider.py``)."""

    def __init__(self, when: Any) -> None:
        super().__init__()
        self._when = when

    def build_estimator_factory(
        self, task: Any, params: dict[str, Any], *args: Any, **kwargs: Any
    ) -> Any:
        if self._when(params):
            return lambda: object()
        return super().build_estimator_factory(task, params, *args, **kwargs)


ONLY_16 = {"num_leaves": {"type": "categorical", "choices": [16]}}


def test_a_foreign_base_value_that_never_runs_is_not_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    train_stub: _Recorder,  # noqa: F811
) -> None:
    # The base num_leaves=8 builds a foreign adapter, but every trial of the
    # round samples 16, which the provider can estimate.
    _use(monkeypatch, _ForeignWhen(lambda p: p.get("num_leaves") == 8))
    cfg = _cfg(space=_space(**ONLY_16), auto=False, n_trials=2, num_leaves=8)
    url = _url(tmp_path)
    result = Model(cfg).tune(
        data=_df("regression"),
        storage=url,
        study_name="s",
        memory_budget_bytes=BUDGET,
    )
    assert result.best_params["num_leaves"] == 16
    assert len(train_stub.calls) == 2 * N_FOLDS
    assert _end_record(url).status == "completed"


def test_a_foreign_sampled_choice_is_refused_before_any_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    train_stub: _Recorder,  # noqa: F811
) -> None:
    # Only the sampled choice 16 builds a foreign adapter; the base 8 does not.
    _use(monkeypatch, _ForeignWhen(lambda p: p.get("num_leaves") == 16))
    cfg = _cfg(space=_space(**ONLY_16), auto=False, n_trials=2, num_leaves=8)
    url = _url(tmp_path)
    model = Model(cfg)
    with pytest.raises(LizyMLError) as err:
        model.tune(
            data=_df("regression"),
            storage=url,
            study_name="s",
            memory_budget_bytes=BUDGET,
        )
    assert err.value.code is ErrorCode.CONFIG_INVALID
    assert err.value.context["reason"] == "memory_estimate_unsupported"
    _no_study_written(url)  # neither the study nor the round record exists
    assert model._study is None
    assert train_stub.calls == []


def test_a_foreign_adapter_off_the_candidate_points_is_refused_per_trial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    train_stub: _Recorder,  # noqa: F811
) -> None:
    # The documented bound: only the envelope's candidate points are probed. A
    # "none" dimension contributes its lower end alone, so a foreign adapter
    # at any other sampled value is refused per trial as assessment_failed.
    low = _space()["learning_rate"]["low"]
    _use(monkeypatch, _ForeignWhen(lambda p: p.get("learning_rate", low) != low))
    url = _url(tmp_path)
    with pytest.raises(LizyMLError) as err:
        Model(_cfg(space=_space(), auto=False, n_trials=2)).tune(
            data=_df("regression"),
            storage=url,
            study_name="s",
            memory_budget_bytes=BUDGET,
        )
    assert err.value.code is ErrorCode.TUNING_FAILED
    assert err.value.context["memory_refused_this_call"] == 2
    assert "reason=assessment_failed" in err.value.context["first_error"]
    assert train_stub.calls == []


def test_a_candidate_value_that_cannot_be_built_is_left_to_the_trial_guard(
    monkeypatch: pytest.MonkeyPatch,
    train_stub: _Recorder,  # noqa: F811
) -> None:
    # The probe's candidate values come from the provider's memory_role. A
    # failure there is not memory_estimate_unsupported, so admission lets the
    # round start, as for a failed point (round 1's rule, kept).
    def broken(self: LGBMProvider, name: str) -> Any:
        raise TypeError("memory_role failed")

    monkeypatch.setattr(LGBMProvider, "memory_role", broken)
    result = Model(_cfg(space=_space(), auto=False, n_trials=2)).tune(
        data=_df("regression"), memory_budget_bytes=BUDGET
    )
    assert result.best_params
    assert len(train_stub.calls) == 2 * N_FOLDS


# --- round 2 finding 2: the device capability is separate from part A's -----------


def _same_estimates(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _cfg(space=_space(**LEAVES_8_64), auto=False)
    df = _df("regression")
    native = (Model(cfg).estimate_memory(df), Model(cfg).estimate_tune_memory(df))
    for module in ("_model_memory", "_model_tune_memory", "_tune_planning"):
        monkeypatch.setattr(
            f"lizyml.core.{module}.get_provider", lambda _c: _PartAOnly()
        )
    assert Model(cfg).estimate_memory(df) == native[0]
    assert Model(cfg).estimate_tune_memory(df) == native[1]


LEAVES_8_64 = {"num_leaves": {"type": "categorical", "choices": [8, 64]}}


def test_both_estimates_accept_a_provider_with_only_part_a_methods(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lizyml.estimators.provider import MemoryEstimateProvider

    assert isinstance(_PartAOnly(), MemoryEstimateProvider)
    _same_estimates(monkeypatch)


def test_the_part_a_only_fixture_lacks_the_device_capability() -> None:
    from lizyml.estimators.provider import TrialDeviceProvider

    assert not isinstance(_PartAOnly(), TrialDeviceProvider)
    assert isinstance(LGBMProvider(), TrialDeviceProvider)


def test_a_budget_needs_the_device_capability_before_any_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    train_stub: _Recorder,  # noqa: F811
) -> None:
    _use(monkeypatch, _PartAOnly())
    url = _url(tmp_path)
    model = Model(_cfg(space=_space(), n_trials=2))
    with pytest.raises(LizyMLError) as err:
        model.tune(
            data=_df("regression"),
            storage=url,
            study_name="s",
            memory_budget_bytes=BUDGET,
        )
    assert err.value.code is ErrorCode.CONFIG_INVALID
    assert err.value.context["reason"] == "memory_estimate_unsupported"
    _no_study_written(url)
    assert model._study is None
    assert train_stub.calls == []


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
    assert _end_record(url) is None  # the round is open
    _no_budget_in(tmp_path / "study.db", BUDGET)
    result = Model(cfg).tune(
        data=df,
        resume=True,
        storage=url,
        study_name="s",
        memory_budget_bytes=resume_budget,
    )
    assert result.best_params and len(train_stub.calls) > 0
    _no_budget_in(tmp_path / "study.db", BUDGET, BUDGET // 7)


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
    assert _end_record(url) is None  # open, with no trial left to run
    _no_budget_in(tmp_path / "study.db")  # nothing recorded while open
    trials_before = _trial_tables(tmp_path / "study.db")
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
    # Resume closed the existing round as failed and added no trial.
    end = _end_record(url)
    assert end is not None and end.status == "failed"
    assert _trial_tables(tmp_path / "study.db") == trials_before


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


# --- round 3 finding 1: an unprintable candidate error does not stop the probe ------


class _UnprintableError(TypeError):
    def __str__(self) -> str:
        raise KeyError("a misbehaving __str__")


class _UnprintableThen(LGBMProvider):
    """``num_leaves=8`` fails to build with an error whose ``__str__`` raises;
    ``num_leaves=16`` builds a foreign adapter when *foreign*, else LightGBM's."""

    def __init__(self, *, foreign: bool) -> None:
        super().__init__()
        self._foreign = foreign

    def build_estimator_factory(
        self, task: Any, params: dict[str, Any], *args: Any, **kwargs: Any
    ) -> Any:
        if params.get("num_leaves") == 8:
            raise _UnprintableError()
        if self._foreign and params.get("num_leaves") == 16:
            return lambda: object()
        return super().build_estimator_factory(task, params, *args, **kwargs)


#: The unprintable choice comes first, so its failure is met before the second.
UNPRINTABLE_FIRST = {"num_leaves": {"type": "categorical", "choices": [8, 16]}}


def _use_everywhere(monkeypatch: pytest.MonkeyPatch, provider: Any) -> None:
    """Hand *provider* to ``tune`` and to ``estimate_tune_memory``."""
    for module in ("_model_tuning", "_tune_planning", "_model_tune_memory"):
        monkeypatch.setattr(f"lizyml.core.{module}.get_provider", lambda _c: provider)


def test_an_unprintable_candidate_error_does_not_hide_an_unsupported_adapter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    train_stub: _Recorder,  # noqa: F811
) -> None:
    _use_everywhere(monkeypatch, _UnprintableThen(foreign=True))
    cfg = _cfg(space=_space(**UNPRINTABLE_FIRST), auto=False, n_trials=2)
    url = _url(tmp_path)
    model = Model(cfg)
    with pytest.raises(LizyMLError) as err:
        model.tune(
            data=_df("regression"),
            storage=url,
            study_name="s",
            memory_budget_bytes=BUDGET,
        )
    assert err.value.code is ErrorCode.CONFIG_INVALID
    assert err.value.context["reason"] == "memory_estimate_unsupported"
    _no_study_written(url)
    assert model._study is None
    assert train_stub.calls == []
    # Part A's rule, through the same loop: the unsupported refusal propagates.
    with pytest.raises(LizyMLError) as est_err:
        Model(cfg).estimate_tune_memory(_df("regression"))
    assert est_err.value.code is ErrorCode.CONFIG_INVALID
    assert est_err.value.context["reason"] == "memory_estimate_unsupported"


def test_an_unprintable_candidate_error_is_recorded_as_a_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use_everywhere(monkeypatch, _UnprintableThen(foreign=False))
    cfg = _cfg(space=_space(**UNPRINTABLE_FIRST), auto=False)
    est = Model(cfg).estimate_tune_memory(_df("regression"))
    assert est.status == "incomplete"
    assert est.failures == ("_UnprintableError: <unprintable message>",)
    assert est.n_candidates == 1  # the second choice was still evaluated
    assert "num_leaves" in est.unresolved


# --- round 3 finding 3: a non-callable finalized_device -------------------------------


class _DeviceNotCallable(LGBMProvider):
    """``finalized_device`` is an attribute, not a method: ``isinstance``
    against the runtime protocol accepts it on Python 3.12."""

    finalized_device = False  # type: ignore[assignment]


def test_a_non_callable_finalized_device_is_refused_before_any_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    train_stub: _Recorder,  # noqa: F811
) -> None:
    _use(monkeypatch, _DeviceNotCallable())
    url = _url(tmp_path)
    model = Model(_cfg(space=_space(), n_trials=2))
    with pytest.raises(LizyMLError) as err:
        model.tune(
            data=_df("regression"),
            storage=url,
            study_name="s",
            memory_budget_bytes=BUDGET,
        )
    assert err.value.code is ErrorCode.CONFIG_INVALID
    assert err.value.context["reason"] == "memory_estimate_unsupported"
    _no_study_written(url)
    assert model._study is None
    assert train_stub.calls == []
