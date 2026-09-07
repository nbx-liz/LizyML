"""A tuning dimension whose name LightGBM does not know is sampled and discarded.

Optuna samples the dimension, the value is forwarded to ``lgb.train``, and
LightGBM drops it because the name is not one it defines. The trial completes,
a score comes back, and the dimension influenced nothing. Every trial in the
study is really a trial of the remaining dimensions -- which is invisible from
the outside, because a study that explores a meaningless axis looks exactly like
one that explores a meaningful one.

The gate (H-0093) rejects such a name at ``Model(...)`` construction. This file
is its population: three name classes x three ``category`` values x three tasks.
The ``category`` axis matters because only ``model`` names reach LightGBM's
parameter space -- ``smart`` and ``training`` names are LizyML's own, and
rejecting those would be a false positive.

Two things are asserted for the rejected cells, not one: that the error is
``CONFIG_INVALID``, and that the name **never reaches** ``lgb.train``. The
second is what the issue is actually about; a gate that raises but still lets a
later path forward the name would satisfy the first alone.
"""

from __future__ import annotations

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
    # A real LightGBM parameter: always fine under category: model.
    "num_leaves": {"accepted_as_model": True, "is_smart": False},
    # A LizyML smart parameter: not a LightGBM name, so wrong under
    # category: model -- and the diagnostic should say which category it wants.
    "num_leaves_ratio": {"accepted_as_model": False, "is_smart": True},
    # Neither: a typo or an invention.
    "not_a_lightgbm_parameter": {"accepted_as_model": False, "is_smart": False},
}

CELLS = [
    pytest.param(name, category, task, id=f"{name}-{category}-{task}")
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


def _config_with_space(task: str, name: str, category: str) -> dict[str, Any]:
    cfg = make_config(task, n_estimators=5, n_splits=2, tuning_n_trials=2)
    cfg["tuning"]["optuna"]["space"] = {
        name: {"type": "int", "low": 4, "high": 8, "category": category}
        if name == "num_leaves"
        else {"type": "float", "low": 0.1, "high": 0.9, "category": category}
    }
    return cfg


@pytest.mark.parametrize(("name", "category", "task"), CELLS)
def test_search_space_name_is_gated(name: str, category: str, task: str) -> None:
    """Reject an unknown ``category: model`` name; leave every other cell alone."""
    cfg = _config_with_space(task, name, category)
    should_reject = category == "model" and not NAMES[name]["accepted_as_model"]

    if not should_reject:
        # Tuning must not raise for a name this gate has no business judging --
        # a smart or training dimension, or a real LightGBM one.
        Model(cfg, data=_df_for(task)).tune()
        return

    with pytest.raises(LizyMLError) as exc:
        Model(cfg, data=_df_for(task)).tune()
    err = exc.value
    assert err.code is ErrorCode.CONFIG_INVALID, (
        f"expected CONFIG_INVALID for {name!r} under category={category!r}, "
        f"got {err.code}"
    )
    assert name in str(err), f"the message must name the offending dimension: {err}"
    if NAMES[name]["is_smart"]:
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
