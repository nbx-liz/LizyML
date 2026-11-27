"""H-0150 part A: the candidate points of ``Model.estimate_tune_memory``.

Policies 2, 3 and 4 (acceptance 2, 3 and 7), as Codex implementation review
round 1 read them:

- a provider that cannot estimate the adapter its factory builds refuses the
  whole estimate (policy 3), never a recorded failure;
- a categorical dimension enumerates every choice and is still asked for its
  memory role: a name outside the provider's closed set is ``unresolved``
  (policy 4, H-0150's ``histogram_pool_size`` example);
- the extra unlimited-depth candidate is LightGBM knowledge, so the provider
  supplies it, for ``max_depth`` only, whenever the range reaches 0 or below
  (policy 2);
- every failed point is its own ``failures`` entry, and each names the
  dimensions that supplied end values or choices at it (policy 4);
- the envelope is the estimate of a trial at the upper grid point (acceptance 2).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import lightgbm
import optuna
import pytest

from lizyml import Model
from lizyml.core import _tune_memory_envelope
from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.estimators.lgbm.provider import LGBMProvider
from tests.test_core.test_h0150_estimate_tune_memory import (
    _cfg,
    _df,
    _estimate,
    _Recorder,
    _space,
    _trial_estimates,
)


@pytest.fixture
def train_stub(monkeypatch: pytest.MonkeyPatch) -> _Recorder:
    recorder = _Recorder()
    monkeypatch.setattr(lightgbm, "train", recorder)
    return recorder


def _points(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Record the trial parameters of every candidate point assessed."""
    seen: list[dict[str, Any]] = []
    real = _tune_memory_envelope.estimate_trial

    def spy(provider: Any, dims: Any, trial_params: dict[str, Any], **kw: Any) -> Any:
        seen.append(dict(trial_params))
        return real(provider, dims, trial_params, **kw)

    monkeypatch.setattr(_tune_memory_envelope, "estimate_trial", spy)
    return seen


def _use(monkeypatch: pytest.MonkeyPatch, provider: LGBMProvider) -> None:
    """Hand *provider* to every module the estimates resolve it in."""
    for module in ("_model_memory", "_model_tune_memory", "_tune_planning"):
        monkeypatch.setattr(f"lizyml.core.{module}.get_provider", lambda _c: provider)


# --- policy 3: an adapter the provider cannot estimate --------------------------


class _ForeignAdapterProvider(LGBMProvider):
    """A capable provider whose factory builds an adapter it cannot estimate."""

    def build_estimator_factory(self, *args: Any, **kwargs: Any) -> Callable[[], Any]:
        return lambda: object()


def test_an_unsupported_adapter_refuses_the_tune_estimate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use(monkeypatch, _ForeignAdapterProvider())
    model = Model(_cfg(space=_space()))
    for call in (
        lambda: model.estimate_tune_memory(_df("regression")),
        lambda: model.estimate_memory(_df("regression")),
    ):
        with pytest.raises(LizyMLError) as err:
            call()
        assert err.value.code is ErrorCode.CONFIG_INVALID
        assert err.value.context["reason"] == "memory_estimate_unsupported"


# --- policy 4: categorical dimensions and the closed set ------------------------


def test_a_categorical_name_outside_the_closed_set_is_unresolved() -> None:
    space = _space(
        histogram_pool_size={"type": "categorical", "choices": [64, 256]},
    )
    est = _estimate(_cfg(space=space), _df("regression"))
    assert est.status == "incomplete"
    assert est.unresolved == ("histogram_pool_size",)
    assert est.n_candidates == 2  # every choice is still evaluated
    assert est.failures == () and est.envelope is not None


def test_a_categorical_name_in_the_closed_set_is_resolved() -> None:
    space = _space(n_estimators={"type": "categorical", "choices": [3, 6]})
    est = _estimate(_cfg(space=space), _df("regression"))
    assert est.status == "complete" and est.n_candidates == 2


@pytest.mark.parametrize("name", ["objective", "device", "device_type"])
def test_the_names_the_estimate_must_know_are_classified(name: str) -> None:
    provider = LGBMProvider()
    assert provider.memory_role(name) is not None


#: Parameters the assessment reads that a categorical dimension can carry
#: (each admitted by tune's registration checks), aliases included.
ASSESSED_CHOICES = [
    "linear_tree",
    "linear_trees",
    "num_class",
    "num_classes",
    "categorical_feature",
    "cat_feature",
    "forcedbins_filename",
    "auto_num_leaves",
]


@pytest.mark.parametrize("name", ASSESSED_CHOICES)
def test_an_assessed_parameter_is_in_the_closed_set(name: str) -> None:
    assert LGBMProvider().memory_role(name) == "choices"


@pytest.mark.parametrize("name", ["linear_tree", "linear_trees"])
def test_a_categorical_linear_tree_is_complete_at_its_true_point(name: str) -> None:
    df = _df("regression")
    space = _space(**{name: {"type": "categorical", "choices": [False, True]}})
    est = _estimate(_cfg(space=space), df)
    assert est.status == "complete" and est.unresolved == ()
    assert est.n_candidates == 2 and est.envelope is not None
    on = _estimate(_cfg(space=_space(), linear_tree=True), df).envelope
    off = _estimate(_cfg(space=_space(), linear_tree=False), df).envelope
    assert on is not None and off is not None
    assert on.estimate_bytes > off.estimate_bytes  # the True point is the larger
    assert est.envelope == on


# --- policy 2: the unlimited-depth candidate ------------------------------------


@pytest.mark.parametrize(
    ("low", "high"),
    [(-1, 4), (-3, -1), (0, 4), (-2, 0)],
    ids=["crosses-zero", "below-zero", "zero-low", "zero-high"],
)
def test_a_depth_range_reaching_zero_evaluates_zero(
    low: int, high: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = _points(monkeypatch)
    space = _space(max_depth={"type": "int", "low": low, "high": high})
    est = _estimate(_cfg(space=space), _df("regression"))
    depths = [p["max_depth"] for p in seen]
    assert sorted(set(depths)) == sorted({low, high, 0})
    assert len(depths) == len(set(depths))  # each value once
    assert est.status == "complete" and est.n_candidates == len(depths)
    assert est.envelope is not None and est.envelope.num_leaves == 131072


def test_a_positive_depth_range_evaluates_its_ends_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = _points(monkeypatch)
    space = _space(max_depth={"type": "int", "low": 3, "high": 6})
    _estimate(_cfg(space=space), _df("regression"))
    assert sorted(p["max_depth"] for p in seen) == [3, 6]


def test_another_increasing_range_gets_no_extra_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = _points(monkeypatch)
    space = _space(num_leaves={"type": "int", "low": -2, "high": 4})
    est = _estimate(_cfg(space=space, auto=False), _df("regression"))
    assert sorted(p["num_leaves"] for p in seen) == [-2, 4]
    assert len(est.failures) == 1  # -2 leaves cannot be evaluated
    assert est.n_candidates == 1 and est.unresolved == ("num_leaves",)


class _NoExtraProvider(LGBMProvider):
    """A provider that supplies no extra candidate for any dimension."""

    def memory_candidates(self, name: str, low: Any, high: Any) -> tuple[Any, ...]:
        return ()


def test_the_extra_candidate_comes_from_the_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use(monkeypatch, _NoExtraProvider())
    seen = _points(monkeypatch)
    space = _space(max_depth={"type": "int", "low": -1, "high": 4})
    _estimate(_cfg(space=space), _df("regression"))
    assert sorted(p["max_depth"] for p in seen) == [-1, 4]


# --- policy 4: failures ----------------------------------------------------------


def test_each_failed_point_is_its_own_failure() -> None:
    """Two points that fail with the same exception are two entries."""
    space = _space(
        early_stopping_rounds={
            "type": "categorical",
            "choices": [None],
            "category": "training",
        },
        n_estimators={"type": "categorical", "choices": [3, 6]},
    )
    est = _estimate(_cfg(space=space), _df("regression"))
    assert len(est.failures) == 2
    assert all(f.startswith("TypeError") for f in est.failures)
    assert est.n_candidates == 0 and est.envelope is None


def test_a_failing_single_choice_is_unresolved() -> None:
    space = _space(
        early_stopping_rounds={
            "type": "categorical",
            "choices": [None],
            "category": "training",
        }
    )
    est = _estimate(_cfg(space=space), _df("regression"))
    assert est.status == "incomplete" and len(est.failures) == 1
    assert est.unresolved == ("early_stopping_rounds",)


# --- acceptance 2: the upper grid point --------------------------------------------


def test_a_trial_at_the_upper_grid_point_equals_the_envelope(
    monkeypatch: pytest.MonkeyPatch, train_stub: _Recorder
) -> None:
    # A log int dimension with a step is sampled as a raw float and rounded by
    # `grid_value` (`suggest_on_grid`), so the trial reaches 64 through the grid.
    num_leaves = {"type": "int", "low": 16, "high": 64, "step": 16, "log": True}
    cfg = _cfg(space=_space(num_leaves=num_leaves), auto=False, n_trials=1)
    df = _df("regression")
    est = _estimate(cfg, df)
    assert est.status == "complete" and est.envelope is not None
    # Force the one trial's raw sample off the grid, next to the upper end.
    real = optuna.trial.Trial.suggest_float

    def suggest_float(self: Any, name: str, *args: Any, **kwargs: Any) -> float:
        if name == "num_leaves":
            return 63.0
        return real(self, name, *args, **kwargs)

    monkeypatch.setattr(optuna.trial.Trial, "suggest_float", suggest_float)
    Model(cfg).tune(data=df)
    assert train_stub.calls[0][0]["num_leaves"] == 64  # the grid point trained
    sizes = _trial_estimates(train_stub, df.drop(columns="target"), auto=False)
    assert sizes == [est.envelope.estimate_bytes]
