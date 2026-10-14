"""H-0120 amendment 2: a regression config cannot stratify on the target.

Stratification keeps class proportions; a regression target has no classes.
Each of the four config positions that stratify on the target is refused with
``CONFIG_INVALID`` before training, by ``Model.fit`` and ``Model.tune``, for a
continuous target and for an integer-valued one (which scikit-learn would
otherwise treat as class labels). The same positions stay accepted for
classification.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from lizyml import Model
from lizyml.core.exceptions import ErrorCode, LizyMLError
from tests._train_spy import record_lightgbm_calls
from tests.test_codegen._retrain_harness import make_config, make_frame

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

#: position -> (config builder, the position name the message must carry,
#: the non-stratified alternative it must offer)
POSITIONS: dict[str, tuple[Any, str, str]] = {
    "stratified_kfold": (
        lambda task: make_config(task, "stratified_kfold"),
        "split.method='stratified_kfold'",
        "kfold",
    ),
    "stratified_group_kfold": (
        lambda task: make_config(task, "stratified_group_kfold"),
        "split.method='stratified_group_kfold'",
        "group_kfold",
    ),
    "blocked_groups_stratify": (
        lambda task: make_config(
            task,
            "blocked_group_kfold",
            split_extra={
                "groups": {"col": "g", "n_splits": 2, "shuffle": True, "stratify": True}
            },
        ),
        "split.groups.stratify=true",
        "stratify: false",
    ),
    "inner_valid_stratify": (
        lambda task: make_config(
            task,
            "kfold",
            early_stopping={"inner_valid": {"method": "holdout", "stratify": True}},
        ),
        "training.early_stopping.inner_valid.stratify=true",
        "stratify: false",
    ),
}

TARGETS = ("continuous", "integer-valued")


def _regression_frame(target: str) -> pd.DataFrame:
    df = make_frame("regression")
    if target == "integer-valued":
        # Few distinct integers, each well populated: scikit-learn would
        # stratify these as classes.
        df["target"] = np.clip(np.round(df["target"]), -2, 2).astype(float)
    return df


def _with_tuning(cfg: dict[str, Any]) -> dict[str, Any]:
    space = {"num_leaves": {"type": "categorical", "choices": [7]}}
    return {
        **cfg,
        "tuning": {
            "optuna": {
                "params": {"n_trials": 1, "direction": "minimize"},
                "space": space,
                "space_mode": "replace",
            }
        },
    }


@pytest.mark.parametrize("call", ["fit", "tune"])
@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("position", sorted(POSITIONS))
def test_regression_stratification_is_refused_before_training(
    position: str, target: str, call: str
) -> None:
    build, name, alternative = POSITIONS[position]
    model = Model(_with_tuning(build("regression")))
    df = _regression_frame(target)
    with record_lightgbm_calls() as seen, pytest.raises(LizyMLError) as info:
        getattr(model, call)(data=df)
    assert info.value.code == ErrorCode.CONFIG_INVALID
    message = info.value.user_message
    assert name in message
    assert alternative in message
    assert seen["train_params"] == []


@pytest.mark.parametrize("task", ["binary", "multiclass"])
@pytest.mark.parametrize("position", sorted(POSITIONS))
def test_classification_keeps_every_position(position: str, task: str) -> None:
    build, _, _ = POSITIONS[position]
    model = Model(build(task))
    model.fit(data=make_frame(task))
    assert model.fit_result is not None


def test_auto_blocked_stratify_is_not_refused_for_regression() -> None:
    """``stratify: auto`` already resolves to no stratification for regression."""
    model = Model(make_config("regression", "blocked_group_kfold"))
    model.fit(data=make_frame("regression"))
    assert model.fit_result is not None
