"""A tuning dimension whose name LightGBM does not know is sampled and discarded.

Optuna samples the dimension, the value is forwarded to ``lgb.train``, and
LightGBM drops it because the name is not one it defines. The trial completes,
a score comes back, and the dimension influenced nothing. Every trial in the
study is really a trial of the remaining dimensions -- which is invisible from
the outside, because a study that explores a meaningless axis looks exactly like
one that explores a meaningful one.

The gate (H-0093) rejects such a name at the entry points that train
(``fit`` / ``tune``), before any training; ``Model(...)`` construction still
succeeds (H-0093 decision 4). This file is its population: three name classes x
three ``category`` values x three tasks. The ``category`` axis matters because
only ``model`` names reach LightGBM's parameter space -- ``smart`` and
``training`` names are LizyML's own. H-0099 additionally refuses training names
without a consumer. Native model names in this matrix explicitly disable the
conflicting smart owner (``auto_num_leaves``); smart cells enable it, because
``num_leaves_ratio`` is consumed only through it.

Every cell records the params dicts actually handed to ``lgb.train`` (#262):
a rejected cell must start no training at all, and an accepted cell must show
its dimension's effect in what LightGBM received -- the dimension's own name for
a native parameter, the resolved native parameter for a smart one. Asserting
only acceptance would pass a dimension that is sampled and then discarded,
which is the defect this file exists for. The smart cells for names that are
not smart parameters are exactly that today, and are strict xfails on #299.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import lightgbm as lgb
import pytest

from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.core.model import Model
from tests._helpers import (
    make_binary_df,
    make_config,
    make_multiclass_df,
    make_regression_df,
)

TASKS = ("regression", "binary", "multiclass")
CATEGORIES = ("model", "smart", "training")

#: The three name classes. ``accepted_as_model`` says whether the gate should
#: let the name through when it is declared with ``category: model``.
NAMES: dict[str, dict[str, Any]] = {
    # A real LightGBM parameter when its smart owner is disabled.
    "num_leaves": {"accepted_as_model": True, "is_smart": False},
    # A LizyML smart parameter: not a LightGBM name, so wrong under
    # category: model -- and the diagnostic should say which category it wants.
    "num_leaves_ratio": {"accepted_as_model": False, "is_smart": True},
    # Neither: a typo or an invention.
    "not_a_lightgbm_parameter": {"accepted_as_model": False, "is_smart": False},
}

#: A ``category: smart`` dimension whose name is not a smart parameter is
#: accepted, sampled, reported in ``best_params`` and consumed by nothing.
#: These cells state the intended outcome (refusal) and fail until #299 lands;
#: ``strict`` makes the fix flip them loudly instead of passing silently.
_INERT_SMART = pytest.mark.xfail(
    strict=True,
    reason="#299: a smart-category name with no consumer is accepted and inert",
)


def _cell_marks(name: str, category: str) -> list[pytest.MarkDecorator]:
    if category == "smart" and not NAMES[name]["is_smart"]:
        return [_INERT_SMART]
    return []


CELLS = [
    pytest.param(
        name,
        category,
        task,
        id=f"{name}-{category}-{task}",
        marks=_cell_marks(name, category),
    )
    for name in NAMES
    for category in CATEGORIES
    for task in TASKS
]


def test_the_population_is_the_declared_cross_product() -> None:
    """The cell count must be the product, not a hand-typed number (DC1)."""
    assert len(CELLS) == len(NAMES) * len(CATEGORIES) * len(TASKS) == 27


def _df_for(task: str) -> Any:
    if task == "regression":
        return make_regression_df(n=120)
    if task == "binary":
        return make_binary_df(n=120)
    return make_multiclass_df(n=150)


@contextmanager
def _record_train_params() -> Iterator[list[dict[str, Any]]]:
    """Record every params dict handed to ``lgb.train``, running the real one."""
    seen: list[dict[str, Any]] = []
    real_train = lgb.train

    def spy(params: dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        seen.append(dict(params))
        return real_train(params, *args, **kwargs)

    lgb.train = spy  # type: ignore[assignment]
    try:
        yield seen
    finally:
        lgb.train = real_train  # type: ignore[assignment]


def _config_with_space(
    task: str, name: str, category: str, *, auto_num_leaves: bool | None = None
) -> dict[str, Any]:
    cfg = make_config(task, n_estimators=5, n_splits=2, tuning_n_trials=2)
    if auto_num_leaves is None:
        auto_num_leaves = category == "smart"
    cfg["model"]["auto_num_leaves"] = auto_num_leaves
    cfg["tuning"]["optuna"]["space"] = {
        name: {"type": "int", "low": 4, "high": 8, "category": category}
        if name == "num_leaves"
        else {"type": "float", "low": 0.1, "high": 0.9, "category": category}
    }
    return cfg


def _num_leaves_bounds_from_ratio(max_depth: int | None) -> tuple[int, int]:
    """The ``num_leaves`` range ``auto_num_leaves`` can derive from a ratio in
    the cell's sampled interval [0.1, 0.9] (``smart_params._compute_num_leaves``).
    """
    base = 131072 if max_depth is None or max_depth < 0 else 2**max_depth
    return (
        max(8, min(131072, math.ceil(base * 0.1))),
        max(8, min(131072, math.ceil(base * 0.9))),
    )


def _assert_effect_reached_lightgbm(
    name: str, category: str, seen: list[dict[str, Any]]
) -> None:
    """An accepted dimension must be visible in what LightGBM received."""
    assert seen, "the study trained nothing, so no dimension had an effect"
    if category == "model":
        # A native name reaches LightGBM under its own name, inside the
        # declared range [4, 8], in every training call.
        values = [p.get(name) for p in seen]
        assert all(v is not None and 4 <= v <= 8 for v in values), (
            f"{name!r} under category: model did not reach every lgb.train call "
            f"within its declared range: {values}"
        )
        return
    # A smart parameter never reaches LightGBM under its own name; its effect is
    # the native parameter it resolves to.
    assert name == "num_leaves_ratio", f"no expected effect declared for {name!r}"
    forwarded = [p for p in seen if name in p]
    assert not forwarded, f"the smart name itself reached lgb.train: {forwarded[:1]}"
    for p in seen:
        low, high = _num_leaves_bounds_from_ratio(p.get("max_depth"))
        assert "num_leaves" in p and low <= p["num_leaves"] <= high, (
            "num_leaves_ratio was sampled but lgb.train did not receive a "
            f"num_leaves derived from it (expected {low}..{high}): {p}"
        )


@pytest.mark.parametrize(("name", "category", "task"), CELLS)
def test_search_space_name_is_gated(name: str, category: str, task: str) -> None:
    """Reject unconsumed names before training; prove accepted ones take effect."""
    cfg = _config_with_space(task, name, category)
    should_reject = (
        category == "training"
        or (category == "model" and not NAMES[name]["accepted_as_model"])
        or (category == "smart" and not NAMES[name]["is_smart"])
    )

    if not should_reject:
        with _record_train_params() as seen:
            Model(cfg, data=_df_for(task)).tune()
        _assert_effect_reached_lightgbm(name, category, seen)
        return

    with _record_train_params() as seen, pytest.raises(LizyMLError) as exc:
        Model(cfg, data=_df_for(task)).tune()
    assert not seen, (
        f"{name!r} under category={category!r} was refused only after "
        f"{len(seen)} lgb.train call(s); the refusal must precede all training"
    )
    err = exc.value
    assert err.code is ErrorCode.CONFIG_INVALID, (
        f"expected CONFIG_INVALID for {name!r} under category={category!r}, "
        f"got {err.code}"
    )
    assert name in str(err), f"the message must name the offending dimension: {err}"
    if category == "model" and NAMES[name]["is_smart"]:
        assert "smart" in str(err), (
            f"{name!r} is a smart parameter, so the message should point at "
            f"category: smart rather than only rejecting it. Got: {err}"
        )


@pytest.mark.parametrize("task", TASKS)
def test_rejected_name_never_reaches_lightgbm(task: str) -> None:
    """The point of the gate: the bad name must not be forwarded to lgb.train.

    Asserting only that construction raises would be satisfied by a gate that
    refuses and then lets some other path forward the name anyway.
    """
    cfg = _config_with_space(task, "not_a_lightgbm_parameter", "model")
    with _record_train_params() as seen, pytest.raises(LizyMLError):
        Model(cfg, data=_df_for(task)).tune()
    forwarded = [p for p in seen if "not_a_lightgbm_parameter" in p]
    assert not forwarded, (
        f"the rejected dimension still reached lgb.train in {len(forwarded)} "
        f"call(s): {forwarded[:1]}"
    )


@pytest.mark.parametrize("task", TASKS)
def test_accepted_name_does_reach_lightgbm(task: str) -> None:
    """The negative control: a real name must still get through to the booster.

    Without this, a gate that rejected everything would pass every assertion
    above.
    """
    cfg = _config_with_space(task, "num_leaves", "model")
    with _record_train_params() as seen:
        Model(cfg, data=_df_for(task)).tune()
    assert seen, "no lgb.train call was recorded"
    assert any("num_leaves" in p for p in seen), (
        "num_leaves is a real LightGBM parameter and must still reach lgb.train"
    )


@pytest.mark.xfail(
    strict=True,
    reason="#299: num_leaves_ratio is inert while auto_num_leaves is off",
)
@pytest.mark.parametrize("task", TASKS)
def test_smart_dimension_with_inactive_consumer_is_refused(task: str) -> None:
    """A real smart parameter whose consumer is switched off is inert too.

    ``num_leaves_ratio`` is read only when ``auto_num_leaves`` is on. With it
    off, the dimension is sampled and reported in ``best_params`` while no
    ``lgb.train`` call receives a ``num_leaves`` -- the same silent pass as an
    unknown name, reached through a valid one.
    """
    cfg = _config_with_space(task, "num_leaves_ratio", "smart", auto_num_leaves=False)
    with _record_train_params() as seen, pytest.raises(LizyMLError) as exc:
        Model(cfg, data=_df_for(task)).tune()
    assert not seen
    assert exc.value.code is ErrorCode.CONFIG_INVALID
