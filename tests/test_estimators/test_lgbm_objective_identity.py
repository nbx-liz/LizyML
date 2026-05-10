"""H-0079 Phase 1 — Parametric end-to-end identity guard for LGBM objective.

Layer 1 of the 7-layer regression prevention: for every (task, objective)
pair in TASK_COMPATIBLE_OBJECTIVES, a small fit must produce a booster
whose internal ``params["objective"]`` matches the requested value
bit-for-bit. Guards against:

- silent strip in ``_build_params()`` (the original H-0079 bug)
- reintroduction of forced ``_TASK_OBJECTIVE`` override
- alias collapsing (e.g. ``"mse" -> "regression"``) at the wrong layer

A single failure here means user/Optuna-supplied ``objective`` did not
reach ``lgb.train``, which is the exact regression mode of the original
bug — `tuning_table` reports one value while the booster trains with
another.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lizyml.config.schema import LizyMLConfig
from lizyml.core.model import Model
from tests._helpers import make_binary_df, make_config, make_multiclass_df


def _make_positive_regression_df(n: int = 80, seed: int = 0) -> pd.DataFrame:
    """Regression DataFrame with strictly positive targets.

    Required for ``gamma`` (target > 0), ``poisson`` / ``tweedie`` (target
    >= 0), and ``mape`` (target != 0). Negative-tolerant objectives
    (``regression``, ``regression_l1``, ``huber``, ``fair``, ``quantile``)
    still fit happily on positive data.
    """
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "feat_a": rng.uniform(0, 10, n),
            "feat_b": rng.uniform(0, 5, n),
        }
    )
    df["target"] = df["feat_a"] * 0.5 + df["feat_b"] + rng.uniform(0.1, 1.0, n)
    return df


_TASK_DATA_FACTORY = {
    "regression": lambda: _make_positive_regression_df(n=80, seed=0),
    "binary": lambda: make_binary_df(n=80, seed=0),
    "multiclass": lambda: make_multiclass_df(n=120, seed=0),
}


_TASK_OBJECTIVE_PAIRS = [
    # regression: 9 canonical LightGBM objectives
    ("regression", "regression"),
    ("regression", "regression_l1"),
    ("regression", "huber"),
    ("regression", "fair"),
    ("regression", "poisson"),
    ("regression", "quantile"),
    ("regression", "mape"),
    ("regression", "gamma"),
    ("regression", "tweedie"),
    # binary: 3 canonical LightGBM objectives
    ("binary", "binary"),
    ("binary", "cross_entropy"),
    ("binary", "cross_entropy_lambda"),
    # multiclass: 2 canonical LightGBM objectives
    ("multiclass", "multiclass"),
    ("multiclass", "multiclassova"),
]


class TestUserObjectiveReachesBooster:
    """L1: user-supplied ``objective`` must survive ``_build_params()``."""

    @pytest.mark.parametrize(("task", "objective"), _TASK_OBJECTIVE_PAIRS)
    def test_user_objective_reaches_booster(self, task: str, objective: str) -> None:
        """Every (task, objective) pair in TASK_COMPATIBLE_OBJECTIVES must
        produce a booster whose ``params["objective"]`` matches input."""
        df = _TASK_DATA_FACTORY[task]()
        # Avoid metric/objective coupling regressions — keep metric default.
        cfg_dict = make_config(
            task,
            n_estimators=10,
            n_splits=2,
            objective=objective,
        )
        cfg = LizyMLConfig(**cfg_dict)
        m = Model(cfg)
        m.fit(data=df)

        # Inspect the refit booster (post-fit, full-data).
        refit_booster = m._refit_result.model.get_native_model()  # type: ignore[union-attr]
        actual = refit_booster.params["objective"]
        assert actual == objective, (
            f"Requested objective='{objective}' for task='{task}' was not "
            f"honoured by the refit booster (got '{actual}'). "
            f"_build_params() may be stripping or overriding objective."
        )

        # Also inspect a per-fold booster (CV path uses the same code path
        # but a separate adapter instance).
        fold_booster = m.fit_result.models[0].get_native_model()  # type: ignore[union-attr]
        actual_fold = fold_booster.params["objective"]
        assert actual_fold == objective, (
            f"Requested objective='{objective}' for task='{task}' was not "
            f"honoured by the per-fold booster (got '{actual_fold}')."
        )


class TestCrossTaskObjectiveRejected:
    """Cross-task injection still raises (existing defense, contract preserved)."""

    @pytest.mark.parametrize(
        ("task", "bad_objective"),
        [
            ("regression", "binary"),
            ("regression", "multiclass"),
            ("binary", "regression"),
            ("binary", "huber"),
            ("binary", "multiclass"),
            ("multiclass", "regression"),
            ("multiclass", "binary"),
        ],
    )
    def test_cross_task_objective_raises(self, task: str, bad_objective: str) -> None:
        """Objective from another task must raise ``CONFIG_INVALID``."""
        from lizyml.core.exceptions import LizyMLError

        df = _TASK_DATA_FACTORY[task]()
        cfg_dict = make_config(
            task,
            n_estimators=10,
            n_splits=2,
            objective=bad_objective,
        )
        cfg = LizyMLConfig(**cfg_dict)
        m = Model(cfg)
        with pytest.raises(LizyMLError) as excinfo:
            m.fit(data=df)
        assert excinfo.value.code.name == "CONFIG_INVALID"
        # Context must include enough info for users to diagnose
        ctx = excinfo.value.context
        assert ctx.get("task") == task
        assert ctx.get("objective") == bad_objective
