"""H-0150 part B slice 1: ``Model.tune(memory_budget_bytes=...)``.

Acceptance 6 (the per-trial refusal and its log reasons, the ``TUNING_FAILED``
context keys) and the value and capability halves of acceptance 7. No test
trains a large model: ``lightgbm.train`` is a recording stub, so "never
trained" is read from the stub's call list.
"""

from __future__ import annotations

import dataclasses
import logging
import sys
from pathlib import Path
from typing import Any

import lightgbm
import optuna
import pytest

from lizyml import Model
from lizyml.core import _tune_budget
from lizyml.core._model_tuning import ModelTuningMixin
from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.core.types.memory_estimate import MemoryEstimate
from lizyml.estimators.lgbm.adapter import LGBMAdapter
from lizyml.estimators.lgbm.provider import LGBMProvider
from lizyml.training.cv_trainer import CVTrainer
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
    """Skip the admission's candidate-point estimates, so a test that makes the
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


def test_a_budgeted_trial_builds_its_components_once_and_trains_with_them(
    monkeypatch: pytest.MonkeyPatch,
    train_stub: _Recorder,  # noqa: F811
) -> None:
    # Round 2 finding 3: each successful budgeted trial builds its components
    # exactly once, and the device check, the estimate and the trainer all
    # receive that build's estimator factory (the same object).
    in_round = {"on": False}
    built: list[Any] = []
    device_seen: list[Any] = []
    estimated: list[Any] = []
    trained: list[Any] = []

    real_build = Model._build_train_components

    def build(self: Any, *a: Any, **k: Any) -> Any:
        tc = real_build(self, *a, **k)
        if in_round["on"]:
            built.append(tc.estimator_factory)
        return tc

    real_run = ModelTuningMixin._run_tune_round

    def run(self: Any, *a: Any, **k: Any) -> Any:
        in_round["on"] = True  # the admission's probe builds come before
        return real_run(self, *a, **k)

    real_device = LGBMProvider.finalized_device
    real_estimate = LGBMProvider.estimate_memory

    def device(self: LGBMProvider, **kw: Any) -> Any:
        if in_round["on"]:
            device_seen.append(kw["estimator_factory"])
        return real_device(self, **kw)

    def estimate(self: LGBMProvider, **kw: Any) -> Any:
        if in_round["on"]:
            estimated.append(kw["estimator_factory"])
        return real_estimate(self, **kw)

    real_init = CVTrainer.__init__

    def init(self: Any, *a: Any, **k: Any) -> None:
        trained.append(k["estimator_factory"])
        real_init(self, *a, **k)

    monkeypatch.setattr(Model, "_build_train_components", build)
    monkeypatch.setattr(ModelTuningMixin, "_run_tune_round", run)
    monkeypatch.setattr(LGBMProvider, "finalized_device", device)
    monkeypatch.setattr(LGBMProvider, "estimate_memory", estimate)
    monkeypatch.setattr(CVTrainer, "__init__", init)
    result = _tune(
        _cfg(space=_space(), auto=False, n_trials=3), memory_budget_bytes=BIG
    )
    assert [t.state for t in result.trials] == ["complete"] * 3
    assert len(built) == 3  # one build per trial, no second one for training
    assert len(device_seen) == len(estimated) == len(trained) == 3
    for i in range(3):
        assert device_seen[i] is built[i]
        assert estimated[i] is built[i]
        assert trained[i] is built[i]


# --- round 3: the budget's bound (#423 issuecomment-7220835949) ----------------------

#: The largest budget: ``1 <= memory_budget_bytes <= 2**63 - 1``.
MAX_BUDGET = 2**63 - 1
#: 4301 digits: ``str()`` of it raises ``ValueError`` under Python's default
#: integer-string limit (4300 digits), which ``digit_limit`` pins.
HUGE = 10**4300
BIG = 10**15


@pytest.fixture
def digit_limit() -> Any:
    """Pin Python's integer-string limit to its default (4300) and restore it,
    so a test does not depend on ``PYTHONINTMAXSTRDIGITS`` (round 3 finding 4).
    Python 3.10 before 3.10.7 has no limit and nothing to pin."""
    if not hasattr(sys, "set_int_max_str_digits"):
        yield None
        return
    before = sys.get_int_max_str_digits()
    sys.set_int_max_str_digits(4300)
    try:
        yield 4300
    finally:
        sys.set_int_max_str_digits(before)


class _UnreprValue:
    """A value whose ``repr`` raises, under any digit limit."""

    def __repr__(self) -> str:
        raise KeyError("a misbehaving __repr__")


# Explicit ids: pytest would build an int parameter's id with ``str()``.
@pytest.mark.parametrize(
    "bad",
    [
        pytest.param(MAX_BUDGET + 1, id="2**63"),
        pytest.param(HUGE, id="10**4300"),
        pytest.param(-HUGE, id="-(10**4300)"),
        pytest.param([HUGE], id="[10**4300]"),
        pytest.param(_UnreprValue(), id="unrepr"),
    ],
)
def test_a_budget_outside_the_bound_is_config_invalid(
    bad: Any,
    digit_limit: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    train_stub: _Recorder,  # noqa: F811
) -> None:
    planned: list[int] = []
    real_plan = Model._plan_tune

    def spy(self: Any, *a: Any, **k: Any) -> Any:
        planned.append(1)
        return real_plan(self, *a, **k)

    monkeypatch.setattr(Model, "_plan_tune", spy)
    url = _url(tmp_path)
    with pytest.raises(LizyMLError) as err:
        _tune(
            _cfg(space=_space(), n_trials=2),
            memory_budget_bytes=bad,
            storage=url,
            study_name="s",
        )
    assert err.value.code is ErrorCode.CONFIG_INVALID
    assert err.value.context["reason"] == "invalid_memory_budget"
    assert planned == []  # refused before planning
    assert train_stub.calls == []
    assert not (tmp_path / "study.db").exists()
    assert "memory_budget_bytes" in str(err.value)
    repr(err.value)  # the message and the context are printable


@pytest.mark.parametrize(
    ("value", "text"),
    [
        pytest.param(HUGE, "<int of 14285 bits>", id="10**4300"),
        pytest.param(-HUGE, "<-int of 14285 bits>", id="-(10**4300)"),
        pytest.param([HUGE], "<unprintable list>", id="[10**4300]"),
        pytest.param(_UnreprValue(), "<unprintable _UnreprValue>", id="unrepr"),
        pytest.param(MAX_BUDGET + 1, "9223372036854775808 (int)", id="2**63"),
        pytest.param(1.5, "1.5 (float)", id="float"),
    ],
)
def test_an_invalid_budget_is_rendered_without_raising(
    value: Any, text: str, digit_limit: Any
) -> None:
    with pytest.raises(LizyMLError) as err:
        _tune_budget.validate_budget_value(value)
    assert err.value.context["value"] == text
    assert text in err.value.user_message


def test_the_largest_budget_runs(train_stub: _Recorder) -> None:  # noqa: F811
    assert _tune_budget.validate_budget_value(MAX_BUDGET) == MAX_BUDGET
    result = _tune(
        _cfg(space=_space(), auto=False, n_trials=2), memory_budget_bytes=MAX_BUDGET
    )
    assert result.best_params
    assert len(train_stub.calls) == 2 * N_FOLDS


def test_a_refusal_under_the_largest_budget_is_printable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    train_stub: _Recorder,  # noqa: F811
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Optuna writes repr() of every FAIL trial's error; the round error keeps
    # the budget as an int (round 3 finding 5): within the bound, both print.
    _no_probe(monkeypatch)
    _raising_estimate(monkeypatch, TypeError("outside catch="), first_only=False)
    caplog.set_level(logging.INFO)
    cfg = _cfg(space=_space(), auto=False, n_trials=2)
    url = _url(tmp_path)
    err = _tune_failed(cfg, memory_budget_bytes=MAX_BUDGET, storage=url, study_name="s")
    assert err.context["memory_refused_this_call"] == 2
    assert err.context["memory_budget_bytes"] == MAX_BUDGET  # the value, an int
    assert "budget=9223372036854775807" in err.context["first_error"]
    assert "reason=assessment_failed" in err.context["first_error"]
    repr(err)
    assert _states(url) == ["FAIL", "FAIL"]
    lines = _refusals(caplog)
    assert len(lines) == 2
    assert all("budget=9223372036854775807" in m for m in lines)


class _HostileInt(int):
    """An ``int`` subclass in range whose printing and comparisons raise."""

    def __repr__(self) -> str:
        raise KeyError("a misbehaving __repr__")

    def __str__(self) -> str:
        raise KeyError("a misbehaving __str__")

    def __format__(self, spec: str) -> str:
        raise KeyError("a misbehaving __format__")

    def __le__(self, other: object) -> bool:
        raise KeyError("a misbehaving __le__")


def test_an_int_subclass_budget_is_used_as_a_plain_int(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    train_stub: _Recorder,  # noqa: F811
) -> None:
    # A valid budget always prints: the value is taken as an exact int.
    plain = _tune_budget.validate_budget_value(_HostileInt(BIG))
    assert type(plain) is int and plain == BIG
    _no_probe(monkeypatch)
    _raising_estimate(monkeypatch, TypeError("outside catch="), first_only=False)
    cfg = _cfg(space=_space(), auto=False, n_trials=2)
    url = _url(tmp_path)
    err = _tune_failed(
        cfg, memory_budget_bytes=_HostileInt(BIG), storage=url, study_name="s"
    )
    assert type(err.context["memory_budget_bytes"]) is int
    assert "budget=1000000000000000" in err.context["first_error"]
    repr(err)
    assert _states(url) == ["FAIL", "FAIL"]


def test_the_refusal_formatter_under_the_largest_budget(
    caplog: pytest.LogCaptureFixture,
) -> None:
    budget = _tune_budget.TuneBudget(limit=MAX_BUDGET)
    caplog.set_level(logging.INFO)
    with pytest.raises(LizyMLError) as err:
        _tune_budget._refuse(
            budget,
            "over_budget",
            "estimate_bytes=1 num_leaves=8",
            "the estimate is over the budget",
        )
    assert err.value.code is ErrorCode.TUNING_FAILED
    assert err.value.context == {
        "reason": "over_budget",
        "memory_budget_bytes": MAX_BUDGET,
    }
    assert "reason=over_budget budget=9223372036854775807" in err.value.user_message
    repr(err.value)  # what Optuna logs for a FAIL trial
    assert budget.refused == 1
    assert len(_refusals(caplog)) == 1


def test_a_huge_estimate_is_refused_with_its_bit_length(
    monkeypatch: pytest.MonkeyPatch,
    digit_limit: Any,
    train_stub: _Recorder,  # noqa: F811
) -> None:
    # The bound covers the budget, not the provider's estimate: an estimate
    # too long for str() is still over the budget and must not raise.
    real = LGBMProvider.estimate_memory

    def huge(self: LGBMProvider, **kw: Any) -> MemoryEstimate:
        est = real(self, **kw)
        return dataclasses.replace(est, estimate_bytes=HUGE, num_leaves=-HUGE)

    _no_probe(monkeypatch)
    monkeypatch.setattr(LGBMProvider, "estimate_memory", huge)
    err = _tune_failed(
        _cfg(space=_space(), auto=False, n_trials=2), memory_budget_bytes=BIG
    )
    first = err.context["first_error"]
    assert "reason=over_budget" in first
    assert "estimate_bytes=<int of 14285 bits>" in first
    assert "num_leaves=<-int of 14285 bits>" in first
    assert train_stub.calls == []


def test_int_text_is_decimal_within_64_bits_and_a_bit_length_beyond(
    digit_limit: Any,
) -> None:
    assert _tune_budget.int_text(MAX_BUDGET) == "9223372036854775807"
    assert _tune_budget.int_text(-(2**63)) == "-9223372036854775808"
    assert _tune_budget.int_text(0) == "0"
    assert _tune_budget.int_text(2**64) == "<int of 65 bits>"
    assert _tune_budget.int_text(HUGE) == "<int of 14285 bits>"
    assert _tune_budget.int_text(-HUGE) == "<-int of 14285 bits>"


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
