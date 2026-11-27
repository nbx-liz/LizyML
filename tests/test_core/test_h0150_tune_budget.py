"""H-0150 part B slice 1: ``Model.tune(memory_budget_bytes=...)``.

Acceptance 6 (the per-trial refusal and its log reasons, the ``TUNING_FAILED``
context keys) and the value and capability halves of acceptance 7. No test
trains a large model: ``lightgbm.train`` is a recording stub, so "never
trained" is read from the stub's call list.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import lightgbm
import optuna
import pytest

from lizyml import Model
from lizyml.core import _tune_budget
from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.estimators.lgbm.adapter import LGBMAdapter
from lizyml.estimators.lgbm.provider import LGBMProvider
from tests.test_core.test_h0150_estimate_tune_memory import (
    N_FOLDS,
    _cfg,
    _df,
    _Recorder,
    _space,
    _url,
    train_stub,  # noqa: F401 -- a fixture, used by name
)

LEAVES = {"num_leaves": {"type": "categorical", "choices": [8, 4096]}}


def _tune(cfg: dict[str, Any], **kw: Any) -> Any:
    return Model(cfg).tune(data=_df("regression"), **kw)


def _tune_failed(cfg: dict[str, Any], **kw: Any) -> LizyMLError:
    with pytest.raises(LizyMLError) as err:
        _tune(cfg, **kw)
    assert err.value.code is ErrorCode.TUNING_FAILED
    return err.value


def _refusals(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if "event='tune.trial.memory_refused'" in r.getMessage()
    ]


def _bytes(cfg: dict[str, Any]) -> int:
    est = Model(cfg).estimate_tune_memory(data=_df("regression"))
    assert est.envelope is not None
    return est.envelope.estimate_bytes


# --- acceptance 6: refusal over the budget -----------------------------------------


def test_an_over_budget_trial_is_never_trained_and_is_logged(
    train_stub: _Recorder,  # noqa: F811
    caplog: pytest.LogCaptureFixture,
) -> None:
    cfg = _cfg(space=_space(), auto=False, n_trials=3)
    caplog.set_level(logging.INFO)
    err = _tune_failed(cfg, memory_budget_bytes=1)
    assert train_stub.calls == []
    lines = _refusals(caplog)
    assert len(lines) == 3
    assert all("reason=over_budget" in m and "budget=1" in m for m in lines)
    assert all("num_leaves=" in m and "estimate_bytes=" in m for m in lines)
    assert err.context["memory_refused_this_call"] == 3
    assert err.context["memory_budget_bytes"] == 1


def test_a_refused_trial_is_an_ordinary_fail_in_the_study(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    train_stub: _Recorder,  # noqa: F811
) -> None:
    cfg = _cfg(space=_space(), auto=False, n_trials=2)
    url = _url(tmp_path)
    _tune_failed(cfg, memory_budget_bytes=1, storage=url, study_name="budget")
    refused = optuna.load_study(study_name="budget", storage=url)

    def boom(*a: Any, **k: Any) -> Any:
        raise RuntimeError("the trial failed for another reason")

    monkeypatch.setattr(lightgbm, "train", boom)
    plain_url = f"sqlite:///{tmp_path / 'plain.db'}"
    _tune_failed(cfg, storage=plain_url, study_name="plain")
    plain = optuna.load_study(study_name="plain", storage=plain_url)

    def shape(study: Any) -> list[tuple[str, set[str], set[str]]]:
        return [
            (t.state.name, set(t.user_attrs), set(t.system_attrs)) for t in study.trials
        ]

    assert shape(refused) == shape(plain)
    assert {t.state.name for t in refused.trials} == {"FAIL"}


def test_an_estimate_equal_to_the_budget_passes_and_one_less_is_refused(
    train_stub: _Recorder,  # noqa: F811
) -> None:
    cfg = _cfg(space=_space(), auto=False, n_trials=2)
    equal = _bytes(cfg)
    result = _tune(cfg, memory_budget_bytes=equal)
    assert result.best_params and len(train_stub.calls) == 2 * N_FOLDS
    before = len(train_stub.calls)
    err = _tune_failed(cfg, memory_budget_bytes=equal - 1)
    assert len(train_stub.calls) == before
    assert err.context["memory_refused_this_call"] == 2


def test_a_partly_refused_round_picks_its_best_from_the_complete_trials(
    train_stub: _Recorder,  # noqa: F811
    caplog: pytest.LogCaptureFixture,
) -> None:
    cfg = _cfg(space=_space(**LEAVES), auto=False, n_trials=10)
    small = _bytes(
        _cfg(
            space=_space(num_leaves={"type": "categorical", "choices": [8]}), auto=False
        )
    )
    caplog.set_level(logging.INFO)
    result = _tune(cfg, memory_budget_bytes=small)
    refused = len(_refusals(caplog))
    assert 0 < refused < 10
    assert result.best_params["num_leaves"] == 8
    assert len(train_stub.calls) == (10 - refused) * N_FOLDS
    # `TrialResult.state` is lowercase: each refusal is a "fail", the rest train.
    states = [t.state for t in result.trials]
    assert states.count("fail") == refused
    assert states.count("complete") == 10 - refused


def test_a_plain_no_complete_round_carries_zero_and_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(*a: Any, **k: Any) -> Any:
        raise RuntimeError("the trial failed")

    monkeypatch.setattr(lightgbm, "train", boom)
    err = _tune_failed(_cfg(space=_space(), n_trials=2))
    assert err.context["memory_refused_this_call"] == 0
    assert err.context["memory_budget_bytes"] is None
    assert err.context["trial_states"]  # the existing keys stay


def test_budget_none_trains_every_trial(train_stub: _Recorder) -> None:  # noqa: F811
    cfg = _cfg(space=_space(), n_trials=3)
    _tune(cfg, memory_budget_bytes=None)
    assert len(train_stub.calls) == 3 * N_FOLDS


# --- policy 5: the assessment raises, the device is not the CPU ---


def _raising_estimate(
    monkeypatch: pytest.MonkeyPatch, exc: Exception, *, first_only: bool
) -> None:
    """Make the provider's estimate raise *exc* (on the first call only, or on
    every call): an exception outside Optuna's ``catch=`` tuple."""
    real = LGBMProvider.estimate_memory
    seen = {"n": 0}

    def flaky(self: LGBMProvider, **kw: Any) -> Any:
        seen["n"] += 1
        if not first_only or seen["n"] == 1:
            raise exc
        return real(self, **kw)

    monkeypatch.setattr(LGBMProvider, "estimate_memory", flaky)


def _no_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip the admission's representative estimate, so a test that makes the
    estimate or the build raise reaches the per-trial guard with every call
    (``raising=False``: the probe did not exist before H-0150 round 1)."""
    monkeypatch.setattr(
        _tune_budget, "probe_adapter", lambda *a, **k: None, raising=False
    )


def _states(url: str, name: str = "s") -> list[str]:
    study = optuna.load_study(study_name=name, storage=url)
    return [t.state.name for t in sorted(study.trials, key=lambda t: t.number)]


def test_an_assessment_that_raises_refuses_that_trial_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    train_stub: _Recorder,  # noqa: F811
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    cfg = _cfg(space=_space(), auto=False, n_trials=3)
    url = _url(tmp_path)
    _no_probe(monkeypatch)
    _raising_estimate(monkeypatch, TypeError("outside catch="), first_only=True)
    result = _tune(cfg, memory_budget_bytes=10**15, storage=url, study_name="s")
    lines = _refusals(caplog)
    assert len(lines) == 1 and "reason=assessment_failed" in lines[0]
    assert "TypeError" in lines[0] and "budget=" in lines[0]
    # The refused trial is persisted FAIL (not pruned) and the run went on.
    assert _states(url) == ["FAIL", "COMPLETE", "COMPLETE"]
    assert len(train_stub.calls) == 2 * N_FOLDS  # the other two trained
    assert result.best_params


def test_every_assessment_that_raises_is_counted_and_the_round_runs_on(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    train_stub: _Recorder,  # noqa: F811
) -> None:
    _no_probe(monkeypatch)
    _raising_estimate(monkeypatch, TypeError("outside catch="), first_only=False)
    cfg = _cfg(space=_space(), auto=False, n_trials=3)
    url = _url(tmp_path)
    err = _tune_failed(cfg, memory_budget_bytes=10**15, storage=url, study_name="s")
    assert err.context["memory_refused_this_call"] == 3
    assert _states(url) == ["FAIL", "FAIL", "FAIL"]  # all three trials ran
    assert train_stub.calls == []
    first = err.context["first_error"]
    assert "reason=assessment_failed" in first and "budget=1000000000000000" in first
    assert "TypeError: outside catch=" in first


class _UnprintableError(TypeError):
    def __str__(self) -> str:
        raise KeyError("a misbehaving __str__")


def test_an_assessment_error_that_cannot_be_printed_is_still_a_refusal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    train_stub: _Recorder,  # noqa: F811
    caplog: pytest.LogCaptureFixture,
) -> None:
    _no_probe(monkeypatch)
    _raising_estimate(monkeypatch, _UnprintableError(), first_only=False)
    caplog.set_level(logging.INFO)
    cfg = _cfg(space=_space(), auto=False, n_trials=2)
    url = _url(tmp_path)
    err = _tune_failed(cfg, memory_budget_bytes=10**15, storage=url, study_name="s")
    assert err.context["memory_refused_this_call"] == 2
    assert _states(url) == ["FAIL", "FAIL"]
    lines = _refusals(caplog)
    assert len(lines) == 2
    assert all("_UnprintableError: <unprintable message>" in m for m in lines)


def test_a_component_build_that_raises_under_a_budget_is_assessment_failed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    train_stub: _Recorder,  # noqa: F811
    caplog: pytest.LogCaptureFixture,
) -> None:
    _no_probe(monkeypatch)
    real = Model._build_train_components
    seen = {"n": 0}

    def flaky(self: Any, *a: Any, **k: Any) -> Any:
        seen["n"] += 1
        if seen["n"] == 1:
            raise TypeError("the components could not be built")
        return real(self, *a, **k)

    monkeypatch.setattr(Model, "_build_train_components", flaky)
    caplog.set_level(logging.INFO)
    cfg = _cfg(space=_space(), auto=False, n_trials=2)
    url = _url(tmp_path)
    _tune(cfg, memory_budget_bytes=10**15, storage=url, study_name="s")
    lines = _refusals(caplog)
    assert len(lines) == 1 and "reason=assessment_failed" in lines[0]
    assert "TypeError: the components could not be built" in lines[0]
    assert _states(url) == ["FAIL", "COMPLETE"]


def test_a_trial_whose_finalized_device_is_not_cpu_is_refused(
    monkeypatch: pytest.MonkeyPatch,
    train_stub: _Recorder,  # noqa: F811
    caplog: pytest.LogCaptureFixture,
) -> None:
    # The merged parameters stay CPU and the estimate is in budget; GPU appears
    # only where the adapter finalizes what LightGBM receives (acceptance 6).
    real = LGBMAdapter._build_params

    def gpu(self: LGBMAdapter, *a: Any, **k: Any) -> Any:
        params, *rest = real(self, *a, **k)
        return ({**params, "device_type": "gpu"}, *rest)

    monkeypatch.setattr(LGBMAdapter, "_build_params", gpu)
    caplog.set_level(logging.INFO)
    cfg = _cfg(space=_space(), auto=False, n_trials=2)
    err = _tune_failed(cfg, memory_budget_bytes=10**15)
    assert train_stub.calls == []
    lines = _refusals(caplog)
    assert len(lines) == 2
    assert all("reason=device_not_cpu" in m and "device=gpu" in m for m in lines)
    assert err.context["memory_refused_this_call"] == 2
    first = err.context["first_error"]
    assert "reason=device_not_cpu" in first and "gpu" in first
    assert "budget=1000000000000000" in first


def test_an_over_budget_refusal_message_carries_its_fields(
    train_stub: _Recorder,  # noqa: F811
) -> None:
    cfg = _cfg(space=_space(), auto=False, n_trials=2)
    err = _tune_failed(cfg, memory_budget_bytes=1)
    first = err.context["first_error"]
    assert "reason=over_budget" in first and "budget=1" in first
    assert "estimate_bytes=" in first and "num_leaves=" in first


# --- acceptance 7: the value and the capability -------------------------------------


@pytest.mark.parametrize("bad", [0, -5, True, 1.5, "100"])
def test_a_bad_budget_value_is_refused_before_the_round(
    bad: Any,
    tmp_path: Path,
    train_stub: _Recorder,  # noqa: F811
) -> None:
    cfg = _cfg(space=_space(), n_trials=2)
    url = _url(tmp_path)
    with pytest.raises(LizyMLError) as err:
        _tune(cfg, memory_budget_bytes=bad, storage=url, study_name="s")
    assert err.value.code is ErrorCode.CONFIG_INVALID
    assert train_stub.calls == []
    assert not (tmp_path / "study.db").exists()  # nothing was opened or written


def test_a_provider_without_the_capability_is_refused_before_any_write(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # `isinstance(..., MemoryEstimateProvider)` is False without the methods.
    class Plain:
        def __init__(self) -> None:
            self._p = LGBMProvider()

        def __getattr__(self, name: str) -> Any:
            if name in ("estimate_memory", "memory_role", "device_is_cpu"):
                raise AttributeError(name)
            return getattr(self._p, name)

    monkeypatch.setattr("lizyml.core._model_tuning.get_provider", lambda _c: Plain())
    monkeypatch.setattr("lizyml.core._tune_planning.get_provider", lambda _c: Plain())
    cfg = _cfg(space=_space(), n_trials=2)
    url = _url(tmp_path)
    with pytest.raises(LizyMLError) as err:
        _tune(cfg, memory_budget_bytes=10**9, storage=url, study_name="s")
    assert err.value.code is ErrorCode.CONFIG_INVALID
    assert err.value.context["reason"] == "memory_estimate_unsupported"
    # Planning may open the storage (as a read does); no study is written.
    assert optuna.get_all_study_summaries(storage=url) == []


# --- the provider's finalized device (policy 6 (b)) ---------------------------------


@pytest.mark.parametrize(
    ("params", "expected", "on_cpu"),
    [
        ({}, {}, True),
        ({"device_type": "cpu"}, {"device_type": "cpu"}, True),
        ({"device": "cuda"}, {"device": "cuda"}, False),
    ],
)
def test_the_finalized_device_is_what_the_adapter_hands_lightgbm(
    params: dict[str, Any], expected: dict[str, Any], on_cpu: bool
) -> None:
    provider = LGBMProvider()
    factory = provider.build_estimator_factory("regression", params, None, None, 0)
    device = provider.finalized_device(estimator_factory=factory)
    assert device == expected
    assert provider.device_is_cpu(device) is on_cpu


def test_a_foreign_adapter_has_no_finalized_device() -> None:
    with pytest.raises(LizyMLError) as err:
        LGBMProvider().finalized_device(estimator_factory=lambda: object())
    assert err.value.code is ErrorCode.CONFIG_INVALID
    assert err.value.context["reason"] == "memory_estimate_unsupported"
