"""Seed reproducibility tests.

Verifies that identical config + seed → identical results across all task types.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from lizyml import Model
from tests._helpers import (
    make_binary_df,
    make_config,
    make_multiclass_df,
    make_regression_df,
)

_TASK_DATA: dict[str, Callable[..., pd.DataFrame]] = {
    "regression": make_regression_df,
    "binary": make_binary_df,
    "multiclass": make_multiclass_df,
}


class TestReproducibility:
    @pytest.mark.parametrize("task", ["regression", "binary", "multiclass"])
    def test_oof_identical(self, task: str) -> None:
        df = _TASK_DATA[task]()
        r1 = Model(make_config(task, n_estimators=20)).fit(data=df)
        r2 = Model(make_config(task, n_estimators=20)).fit(data=df)
        # BLUEPRINT promises bit-identical results for the same config + seed
        # within a fixed (num_threads, CPU) env (H-0081); assert exact equality
        # so a real reproducibility regression cannot hide under rounding.
        np.testing.assert_array_equal(r1.oof_pred, r2.oof_pred)

    def test_predict_identical(self) -> None:
        df = make_regression_df()
        X_new = df.drop(columns=["target"]).iloc[:10].reset_index(drop=True)
        m1 = Model(make_config("regression", n_estimators=20))
        m1.fit(data=df)
        m2 = Model(make_config("regression", n_estimators=20))
        m2.fit(data=df)
        p1 = m1.predict(X_new).pred
        p2 = m2.predict(X_new).pred
        np.testing.assert_array_equal(p1, p2)

    def test_metrics_reproducible(self) -> None:
        df = make_regression_df()
        m1 = Model(make_config("regression", n_estimators=20))
        m1.fit(data=df)
        m2 = Model(make_config("regression", n_estimators=20))
        m2.fit(data=df)
        rmse1 = m1.evaluate()["raw"]["oof"]["rmse"]
        rmse2 = m2.evaluate()["raw"]["oof"]["rmse"]
        assert rmse1 == rmse2


class TestSeedSensitivity:
    """Different ``training.seed`` must change results (decoupling guard).

    The bit-equality tests above both use the default seed, so they cannot
    detect a regression where ``training.seed`` stops propagating (H-0080).
    These tests fail if two distinct seeds yield identical OOF — which would
    mean the seed is silently ignored. ``split.random_state=None`` lets the
    outer splitter inherit ``training.seed`` (H-0080).
    """

    @pytest.mark.parametrize("task", ["regression", "binary", "multiclass"])
    def test_different_seed_changes_oof(self, task: str) -> None:
        df = _TASK_DATA[task]()
        cfg_a = make_config(
            task, n_estimators=20, seed=0, split_overrides={"random_state": None}
        )
        cfg_b = make_config(
            task, n_estimators=20, seed=123, split_overrides={"random_state": None}
        )
        r1 = Model(cfg_a).fit(data=df)
        r2 = Model(cfg_b).fit(data=df)
        assert not np.array_equal(r1.oof_pred, r2.oof_pred), (
            "OOF predictions are identical across different training.seed values "
            "— the seed is not propagating to the outer split (H-0080 regression)."
        )

    def test_same_seed_is_bit_identical_with_inherited_split(self) -> None:
        # Sanity counterpart: with split.random_state=None and the same seed,
        # results stay bit-identical (inheritance is deterministic).
        df = make_regression_df()

        def _cfg() -> dict:
            return make_config(
                "regression",
                n_estimators=20,
                seed=7,
                split_overrides={"random_state": None},
            )

        r1 = Model(_cfg()).fit(data=df)
        r2 = Model(_cfg()).fit(data=df)
        np.testing.assert_array_equal(r1.oof_pred, r2.oof_pred)
