"""Seed reproducibility tests.

Verifies that identical config + seed → identical results across all task types.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from lizyml import Model


def _reg_df(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "feat_a": rng.uniform(0, 10, n),
            "feat_b": rng.uniform(-1, 1, n),
        }
    )
    df["target"] = df["feat_a"] * 2.0 + df["feat_b"] + rng.normal(0, 0.1, n)
    return df


def _bin_df(n: int = 200, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "feat_a": rng.uniform(0, 10, n),
            "feat_b": rng.uniform(-1, 1, n),
        }
    )
    df["target"] = (df["feat_a"] > 5).astype(int)
    return df


def _multi_df(n: int = 300, seed: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "feat_a": rng.uniform(0, 10, n),
            "feat_b": rng.uniform(-1, 1, n),
        }
    )
    df["target"] = pd.cut(df["feat_a"], bins=3, labels=[0, 1, 2]).astype(int)
    return df


def _base_cfg(task: str) -> dict:
    return {
        "config_version": 1,
        "task": task,
        "data": {"target": "target"},
        "split": {"method": "kfold", "n_splits": 3, "random_state": 42},
        "model": {"name": "lgbm", "params": {"n_estimators": 20}},
        "training": {"seed": 0},
    }


class TestReproducibility:
    def test_regression_oof_identical(self) -> None:
        df = _reg_df()
        r1 = Model(_base_cfg("regression")).fit(data=df)
        r2 = Model(_base_cfg("regression")).fit(data=df)
        np.testing.assert_array_almost_equal(r1.oof_pred, r2.oof_pred)

    def test_regression_predict_identical(self) -> None:
        df = _reg_df()
        X_new = df.drop(columns=["target"]).iloc[:10].reset_index(drop=True)
        m1 = Model(_base_cfg("regression"))
        m1.fit(data=df)
        m2 = Model(_base_cfg("regression"))
        m2.fit(data=df)
        p1 = m1.predict(X_new).pred
        p2 = m2.predict(X_new).pred
        np.testing.assert_array_almost_equal(p1, p2)

    def test_binary_oof_identical(self) -> None:
        df = _bin_df()
        r1 = Model(_base_cfg("binary")).fit(data=df)
        r2 = Model(_base_cfg("binary")).fit(data=df)
        np.testing.assert_array_almost_equal(r1.oof_pred, r2.oof_pred)

    def test_multiclass_oof_identical(self) -> None:
        df = _multi_df()
        r1 = Model(_base_cfg("multiclass")).fit(data=df)
        r2 = Model(_base_cfg("multiclass")).fit(data=df)
        np.testing.assert_array_almost_equal(r1.oof_pred, r2.oof_pred)

    def test_metrics_reproducible(self) -> None:
        df = _reg_df()
        m1 = Model(_base_cfg("regression"))
        m1.fit(data=df)
        m2 = Model(_base_cfg("regression"))
        m2.fit(data=df)
        rmse1 = m1.evaluate()["raw"]["oof"]["rmse"]
        rmse2 = m2.evaluate()["raw"]["oof"]["rmse"]
        assert rmse1 == rmse2
