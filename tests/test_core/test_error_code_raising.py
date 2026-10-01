"""Every ``ErrorCode`` member is raised by a condition built here (H-0106, #263).

``test_error_code_population.py`` proves only that a ``raise`` statement names
each member; ``if False: raise LizyMLError(code)`` would satisfy it. Here each
member's condition is built and executed, and the raised ``code`` and the
``context`` keys its raise site supplies are asserted. The table's keys must be
exactly the enum, so a new member without a condition fails here.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import patch

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from lizyml.calibration.platt import PlattCalibrator
from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.core.model import Model
from lizyml.core.types.target_encoder import TargetEncoder
from lizyml.data import validators
from lizyml.estimators.lgbm.metric_bridge import _build_feval
from lizyml.metrics import get_metric
from tests._helpers import make_config


def _binary_df(n: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    df["target"] = (df["a"] + rng.normal(scale=0.5, size=n) > 0).astype(int)
    return df


def _regression_df(n: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    df["target"] = df["a"] * 2 + rng.normal(size=n)
    return df


def _fitted(task: str = "binary") -> Model:
    df = _binary_df() if task == "binary" else _regression_df()
    model = Model(make_config(task, n_estimators=5, n_splits=2))
    model.fit(data=df)
    return model


def _tuning_config() -> dict[str, Any]:
    return make_config("regression", n_estimators=5, n_splits=2, tuning_n_trials=1)


def _resume_without_a_study() -> None:
    Model(_tuning_config()).tune(data=_regression_df(), resume=True)


def _optuna_missing() -> None:
    with patch("lizyml.tuning.tuner._optuna", None):
        Model(_tuning_config()).tune(data=_regression_df())


def _multiclass_feval_gets_1d() -> None:
    feval = _build_feval(get_metric("brier"), "multiclass", num_class=3)
    ds = lgb.Dataset(np.zeros((6, 1)), label=np.array([0, 1, 2, 0, 1, 2], dtype=float))
    ds.construct()
    feval(np.full(6, 0.3), ds)


def _with_config(**overrides: Any) -> dict[str, Any]:
    cfg = make_config("binary", n_estimators=5, n_splits=2)
    cfg.update(overrides)
    return cfg


#: member -> (condition, context keys the raise site supplies).
CONDITIONS: dict[ErrorCode, tuple[Callable[[Path], object], set[str]]] = {
    ErrorCode.CONFIG_INVALID: (
        lambda p: Model(_with_config(unknown_key=1)),
        {"validation_errors"},
    ),
    ErrorCode.CONFIG_VERSION_UNSUPPORTED: (
        lambda p: Model(_with_config(config_version=2)),
        {"config_version", "supported"},
    ),
    ErrorCode.DATA_SCHEMA_INVALID: (
        lambda p: _fitted().predict(_binary_df()[["a"]]),
        {"missing_columns"},
    ),
    ErrorCode.LEAKAGE_SUSPECTED: (
        lambda p: validators.validate_time_series_order(
            pd.DataFrame({"t": [3, 1, 2]}), "t", raise_on_violation=True
        ),
        {"time_col"},
    ),
    ErrorCode.LEAKAGE_CONFIRMED: (
        lambda p: validators.validate_group_split(
            pd.Series(["g1", "g1", "g1"]),
            np.array([0]),
            np.array([1, 2]),
            raise_on_violation=True,
        ),
        {"overlapping_groups"},
    ),
    ErrorCode.OPTIONAL_DEP_MISSING: (lambda p: _optuna_missing(), {"package"}),
    ErrorCode.MODEL_NOT_FIT: (
        lambda p: Model(_with_config()).predict(_binary_df()[["a", "b"]]),
        {"method", "task"},
    ),
    ErrorCode.INCOMPATIBLE_COLUMNS: (
        lambda p: _fitted().predict(_binary_df()[["a", "b"]].astype({"a": str})),
        {"columns"},
    ),
    ErrorCode.UNSUPPORTED_TASK: (
        lambda p: _fitted("regression").confusion_matrix(),
        {"task"},
    ),
    ErrorCode.UNSUPPORTED_METRIC: (
        lambda p: Model(_with_config(evaluation={"metrics": ["rmse"]})).fit(
            data=_binary_df()
        ),
        {"metric", "task"},
    ),
    ErrorCode.METRIC_REQUIRES_PROBA: (
        lambda p: get_metric("auc")(
            np.array([0, 1, 0, 1]), np.array([-2.0, 1.5, -0.5, 3.0])
        ),
        {"metric", "reason"},
    ),
    ErrorCode.TUNING_FAILED: (
        lambda p: _resume_without_a_study(),
        {"resume", "round_number"},
    ),
    ErrorCode.EVALUATION_FAILED: (
        lambda p: _multiclass_feval_gets_1d(),
        {"metric", "num_class", "pred_shape"},
    ),
    ErrorCode.CALIBRATION_NOT_SUPPORTED: (
        lambda p: Model(
            make_config("regression", n_estimators=5, n_splits=2, calibration="platt")
        ).fit(data=_regression_df()),
        {"task"},
    ),
    ErrorCode.CALIBRATION_NOT_FITTED: (
        lambda p: PlattCalibrator().predict(np.array([0.1, 0.9])),
        {"calibrator"},
    ),
    ErrorCode.SERIALIZATION_FAILED: (lambda p: _fitted().export(), set()),
    ErrorCode.DESERIALIZATION_FAILED: (lambda p: Model.load(p / "missing"), {"path"}),
    ErrorCode.TARGET_NOT_NUMERIC: (
        lambda p: Model(make_config("regression", n_estimators=5, n_splits=2)).fit(
            data=_regression_df().assign(target=lambda d: d["target"].astype(str))
        ),
        {"dtype", "task"},
    ),
    ErrorCode.TARGET_UNSEEN_LABEL: (
        lambda p: TargetEncoder.fit(pd.Series(["a", "b"]), "binary").transform(
            pd.Series(["a", "c"])
        ),
        {"known", "unseen"},
    ),
}


def test_the_conditions_cover_the_enum() -> None:
    assert set(CONDITIONS) == set(ErrorCode)


@pytest.mark.parametrize("member", sorted(CONDITIONS, key=lambda m: m.value), ids=str)
def test_member_is_raised(member: ErrorCode, tmp_path: Path) -> None:
    condition, context_keys = CONDITIONS[member]
    with pytest.raises(LizyMLError) as exc:
        condition(tmp_path)
    assert exc.value.code == member
    assert context_keys <= set(exc.value.context)
