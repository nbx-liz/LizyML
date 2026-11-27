"""H-0150 part A: ``Model.estimate_memory`` and the provider capability.

Acceptance 1 (the estimate equals what fit assesses, no training, no state
change), 7 (capability refusal) and 8 (``resource_warning`` unchanged).
"""

from __future__ import annotations

import dataclasses
from typing import Any

import lightgbm
import pandas as pd
import pytest

import lizyml.core.model as model_module
from lizyml import Model
from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.core.types import MemoryEstimate
from lizyml.estimators.lgbm.memory_check import estimate_bytes, feature_bins
from lizyml.estimators.lgbm.provider import LGBMProvider
from lizyml.estimators.provider import MemoryEstimateProvider
from tests._helpers import make_config, make_multiclass_df, make_regression_df


class _FitCapture:
    """Wraps ``check_resources`` to record what fit hands the assessment."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.real = model_module.check_resources

    def __call__(self, provider: Any, tc: Any, X: Any, smart: Any, **kw: Any) -> None:
        self.calls.append(
            {"provider": provider, "tc": tc, "X": X, "smart": smart, **kw}
        )
        self.real(provider, tc, X, smart, **kw)


@pytest.fixture
def capture(monkeypatch: pytest.MonkeyPatch) -> _FitCapture:
    cap = _FitCapture()
    monkeypatch.setattr(model_module, "check_resources", cap)
    return cap


def _cfg(
    task: str = "regression", *, auto: bool = True, **model: Any
) -> dict[str, Any]:
    n_splits = model.pop("n_splits", 3)
    cfg = make_config(task, n_splits=n_splits, **model)
    cfg["model"]["auto_num_leaves"] = auto
    return cfg


def _fit_estimate(
    cfg: dict[str, Any],
    capture: _FitCapture,
    df: pd.DataFrame,
    params: dict[str, Any] | None = None,
) -> MemoryEstimate:
    """The estimate fit computes for *cfg*, through fit's own assessment call."""
    Model(cfg).fit(data=df, params=params)
    call = capture.calls[-1]
    return call["provider"].estimate_memory(
        estimator_factory=call["tc"].estimator_factory,
        X=call["X"],
        smart_params=call["smart"],
        n_boosters=call["n_boosters"],
    )


CASES: list[tuple[str, dict[str, Any], dict[str, Any]]] = [
    ("regression", {"max_depth": 6}, {}),
    ("regression", {"max_depth": -1}, {}),
    ("regression", {"max_depth": 0}, {}),
    ("regression", {}, {"num_leaves": 40, "auto": False}),
    ("regression", {}, {"auto": False}),
    ("regression", {"max_depth": 5}, {"linear_tree": True}),
    ("regression", {}, {"max_bin_by_feature": [8, 16]}),
    ("multiclass", {"max_depth": 5}, {}),
    ("binary", {"max_depth": 5}, {}),
]


def _df(task: str) -> pd.DataFrame:
    if task == "multiclass":
        return make_multiclass_df(n=90)
    if task == "binary":
        df = make_regression_df(n=90)
        df["target"] = (df["target"] > df["target"].median()).astype(int)
        return df
    return make_regression_df(n=90)


@pytest.mark.parametrize(("task", "model", "extra"), CASES)
def test_the_estimate_equals_the_one_fit_assesses(
    task: str, model: dict[str, Any], extra: dict[str, Any], capture: _FitCapture
) -> None:
    extra = dict(extra)
    auto = extra.pop("auto", True)
    cfg = _cfg(task, auto=auto, **model)
    df = _df(task)
    expected = _fit_estimate(cfg, capture, df, params=extra or None)
    got = Model(cfg).estimate_memory(data=df, params=extra or None)
    assert isinstance(got, MemoryEstimate)
    assert got == expected
    assert got.n_boosters == 3 + 1


def test_the_alias_of_num_leaves_is_read_as_lightgbm_reads_it(
    capture: _FitCapture,
) -> None:
    cfg = _cfg(auto=False)
    df = _df("regression")
    got = Model(cfg).estimate_memory(data=df, params={"num_leaf": 77})
    assert got.num_leaves == 77 and got.cause == "explicit"
    assert got == _fit_estimate(cfg, capture, df, params={"num_leaf": 77})


def test_the_cause_is_named(capture: _FitCapture) -> None:
    df = _df("regression")
    derived = Model(_cfg(max_depth=4)).estimate_memory(data=df)
    default = Model(_cfg(auto=False)).estimate_memory(data=df)
    assert derived.cause == "derived" and derived.max_depth == 4
    assert default.cause == "default" and default.num_leaves == 31


def test_it_trains_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*a: Any, **k: Any) -> Any:
        raise AssertionError("estimate_memory must not train")

    monkeypatch.setattr(lightgbm, "train", boom)
    monkeypatch.setattr(lightgbm, "Dataset", boom)
    got = Model(_cfg(max_depth=4)).estimate_memory(data=_df("regression"))
    assert got.estimate_bytes > 0


def _state(model: Model) -> dict[str, Any]:
    return {k: repr(v) for k, v in vars(model).items()}


def test_the_model_state_is_unchanged() -> None:
    model = Model(_cfg(max_depth=4))
    before = _state(model)
    model.estimate_memory(data=_df("regression"))
    assert _state(model) == before
    # and a fit afterwards works as without the call
    model.fit(data=_df("regression"))
    assert model._fit_result is not None


def test_the_state_after_a_fit_is_unchanged() -> None:
    model = Model(_cfg(max_depth=4))
    model.fit(data=_df("regression"))
    before = _state(model)
    model.estimate_memory(data=_df("regression"))
    assert _state(model) == before


def test_the_type_is_frozen() -> None:
    est = Model(_cfg()).estimate_memory(data=_df("regression"))
    assert {f.name for f in dataclasses.fields(est)} == {
        "num_leaves",
        "cause",
        "max_depth",
        "estimate_bytes",
        "n_boosters",
    }
    with pytest.raises(dataclasses.FrozenInstanceError):
        est.num_leaves = 1  # type: ignore[misc]


def test_the_lightgbm_provider_has_the_capability() -> None:
    assert isinstance(LGBMProvider(), MemoryEstimateProvider)
    assert LGBMProvider().memory_role("num_leaves") == "increasing"
    assert LGBMProvider().memory_role("learning_rate") == "none"
    assert LGBMProvider().memory_role("histogram_pool_size") is None


def test_a_provider_without_the_capability_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Bare:
        """Only what is needed to get past the capability check."""

    monkeypatch.setattr(
        "lizyml.core._model_memory.get_provider", lambda _cfg: Bare(), raising=True
    )
    with pytest.raises(LizyMLError) as err:
        Model(_cfg()).estimate_memory(data=_df("regression"))
    assert err.value.code is ErrorCode.CONFIG_INVALID
    assert err.value.context["reason"] == "memory_estimate_unsupported"


@pytest.mark.parametrize(
    ("auto", "params"),
    [(True, None), (False, {"num_leaf": 77}), (False, {"n_estimators": 9})],
)
def test_the_estimate_matches_what_lightgbm_receives(
    monkeypatch: pytest.MonkeyPatch,
    auto: bool,
    params: dict[str, Any] | None,
) -> None:
    """Independent of the provider: recompute from the real ``lgb.train`` call."""
    real = lightgbm.train
    seen: list[tuple[dict[str, Any], int]] = []

    def spy(params: dict[str, Any], train_set: Any, num_boost_round: int, **kw: Any):  # type: ignore[no-untyped-def]
        seen.append((dict(params), int(num_boost_round)))
        return real(params, train_set, num_boost_round=num_boost_round, **kw)

    monkeypatch.setattr(lightgbm, "train", spy)
    cfg = _cfg(auto=auto, max_depth=4, n_estimators=7)
    df = _df("regression")
    got = Model(cfg).estimate_memory(data=df, params=params)
    Model(cfg).fit(data=df, params=params)
    received, rounds = seen[0]
    bins = feature_bins(df[["feat_a", "feat_b"]], received)
    leaves = next(
        (v for k, v in received.items() if k in ("num_leaves", "num_leaf")), 31
    )
    assert got.num_leaves == leaves
    assert got.estimate_bytes == estimate_bytes(
        got.num_leaves, bins, rounds, 1, got.n_boosters
    )
    if params and "n_estimators" in params:
        assert rounds == 9


def test_the_per_call_transients_are_put_back() -> None:
    """A category column makes ``_input_categories`` non-empty after data prep."""
    df = _df("regression")
    df["grp"] = pd.Categorical(["a", "b", "c"] * 30)
    model = Model(_cfg(max_depth=4))
    model.estimate_memory(data=df)
    assert model._input_categories == {}
    assert model._block_values is None
    model._input_categories = {"keep": ["x"]}
    model.estimate_memory(data=df)
    assert model._input_categories == {"keep": ["x"]}


def test_fit_with_no_rounds_fails_as_before_in_training() -> None:
    """Pinned against ecec3da: LightGBM's own ValueError, raised in training.

    The memory warning path must not refuse it first (H-0150 keeps the
    unevaluable refusal inside ``estimate_memory`` and the envelope).
    """
    df = _df("regression")
    cfg = _cfg(n_estimators=-1)
    with pytest.raises(ValueError, match="Number of boosting rounds must be greater"):
        Model(cfg).fit(data=df)
    with pytest.raises(LizyMLError) as err:
        Model(cfg).estimate_memory(data=df)
    assert err.value.code is ErrorCode.CONFIG_INVALID
    assert err.value.context["reason"] == "memory_estimate_unevaluable"
