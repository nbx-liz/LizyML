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
        np.testing.assert_array_almost_equal(r1.oof_pred, r2.oof_pred)

    def test_predict_identical(self) -> None:
        df = make_regression_df()
        X_new = df.drop(columns=["target"]).iloc[:10].reset_index(drop=True)
        m1 = Model(make_config("regression", n_estimators=20))
        m1.fit(data=df)
        m2 = Model(make_config("regression", n_estimators=20))
        m2.fit(data=df)
        p1 = m1.predict(X_new).pred
        p2 = m2.predict(X_new).pred
        np.testing.assert_array_almost_equal(p1, p2)

    def test_metrics_reproducible(self) -> None:
        df = make_regression_df()
        m1 = Model(make_config("regression", n_estimators=20))
        m1.fit(data=df)
        m2 = Model(make_config("regression", n_estimators=20))
        m2.fit(data=df)
        rmse1 = m1.evaluate()["raw"]["oof"]["rmse"]
        rmse2 = m2.evaluate()["raw"]["oof"]["rmse"]
        assert rmse1 == rmse2
