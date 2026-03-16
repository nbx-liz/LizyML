"""Tests for Phase 8 — EstimatorAdapter (LGBMAdapter).

Covers:
- Regression / binary / multiclass fit → predict E2E
- predict_proba shape and unavailability for regression
- Feature importance (split / gain)
- early stopping best_iteration
- Reproducibility (seed fixed)
- MODEL_NOT_FIT error before fit
- UNSUPPORTED_TASK error for predict_proba on regression
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.estimators import LGBMAdapter

# ---------------------------------------------------------------------------
# Synthetic datasets
# ---------------------------------------------------------------------------


@pytest.fixture()
def reg_dataset() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame({"a": rng.uniform(0, 10, n), "b": rng.uniform(-1, 1, n)})
    y = pd.Series(X["a"] * 2.0 + X["b"] + rng.normal(0, 0.1, n), name="target")
    return X, y


@pytest.fixture()
def bin_dataset() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(1)
    n = 200
    X = pd.DataFrame({"a": rng.uniform(0, 10, n), "b": rng.uniform(-1, 1, n)})
    y = pd.Series((X["a"] > 5).astype(int), name="target")
    return X, y


@pytest.fixture()
def multi_dataset() -> tuple[pd.DataFrame, pd.Series, int]:
    rng = np.random.default_rng(2)
    n = 300
    X = pd.DataFrame({"a": rng.uniform(0, 10, n), "b": rng.uniform(-1, 1, n)})
    y = pd.Series(pd.cut(X["a"], bins=3, labels=[0, 1, 2]).astype(int), name="target")
    return X, y, 3


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------


class TestLGBMRegression:
    def test_fit_predict_shape(
        self, reg_dataset: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        X, y = reg_dataset
        adapter = LGBMAdapter(task="regression", random_state=0)
        adapter.fit(X, y)
        preds = adapter.predict(X)
        assert preds.shape == (len(X),)
        assert preds.dtype.kind == "f"

    def test_predict_proba_raises(
        self, reg_dataset: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        X, y = reg_dataset
        adapter = LGBMAdapter(task="regression", random_state=0)
        adapter.fit(X, y)
        with pytest.raises(LizyMLError) as exc_info:
            adapter.predict_proba(X)
        assert exc_info.value.code == ErrorCode.UNSUPPORTED_TASK

    def test_reproducibility(self, reg_dataset: tuple[pd.DataFrame, pd.Series]) -> None:
        X, y = reg_dataset
        a = LGBMAdapter(task="regression", random_state=42)
        b = LGBMAdapter(task="regression", random_state=42)
        a.fit(X, y)
        b.fit(X, y)
        np.testing.assert_array_equal(a.predict(X), b.predict(X))

    def test_importance_split(
        self, reg_dataset: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        X, y = reg_dataset
        adapter = LGBMAdapter(task="regression", random_state=0)
        adapter.fit(X, y)
        imp = adapter.importance("split")
        assert set(imp.keys()) == {"a", "b"}
        assert all(isinstance(v, float) for v in imp.values())

    def test_importance_gain(self, reg_dataset: tuple[pd.DataFrame, pd.Series]) -> None:
        X, y = reg_dataset
        adapter = LGBMAdapter(task="regression", random_state=0)
        adapter.fit(X, y)
        imp = adapter.importance("gain")
        assert set(imp.keys()) == {"a", "b"}

    def test_early_stopping(self, reg_dataset: tuple[pd.DataFrame, pd.Series]) -> None:
        X, y = reg_dataset
        train_X, valid_X = X.iloc[:160], X.iloc[160:]
        train_y, valid_y = y.iloc[:160], y.iloc[160:]
        params = {"n_estimators": 500, "learning_rate": 0.1}
        adapter = LGBMAdapter(
            task="regression",
            params=params,
            early_stopping_rounds=20,
            random_state=0,
        )
        adapter.fit(train_X, train_y, valid_X, valid_y)
        # best_iteration should be < n_estimators due to early stopping
        bi = adapter.best_iteration
        assert bi is not None
        assert 0 < bi < 500

    def test_not_fitted_raises(self) -> None:
        adapter = LGBMAdapter(task="regression")
        X = pd.DataFrame({"a": [1.0]})
        with pytest.raises(LizyMLError) as exc_info:
            adapter.predict(X)
        assert exc_info.value.code == ErrorCode.MODEL_NOT_FIT

    def test_get_native_model(
        self, reg_dataset: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        X, y = reg_dataset
        adapter = LGBMAdapter(task="regression", random_state=0)
        adapter.fit(X, y)
        native = adapter.get_native_model()
        assert isinstance(native, lgb.Booster)
        assert hasattr(native, "predict")


# ---------------------------------------------------------------------------
# Binary classification
# ---------------------------------------------------------------------------


class TestLGBMBinary:
    def test_predict_shape(self, bin_dataset: tuple[pd.DataFrame, pd.Series]) -> None:
        X, y = bin_dataset
        adapter = LGBMAdapter(task="binary", random_state=0)
        adapter.fit(X, y)
        preds = adapter.predict(X)
        assert preds.shape == (len(X),)

    def test_predict_proba_shape(
        self, bin_dataset: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        X, y = bin_dataset
        adapter = LGBMAdapter(task="binary", random_state=0)
        adapter.fit(X, y)
        proba = adapter.predict_proba(X)
        assert proba.shape == (len(X), 2)
        # Probabilities must be in [0, 1] and sum to 1 per row
        assert np.allclose(proba.sum(axis=1), 1.0)
        assert (proba >= 0).all() and (proba <= 1).all()

    def test_reproducibility(self, bin_dataset: tuple[pd.DataFrame, pd.Series]) -> None:
        X, y = bin_dataset
        a = LGBMAdapter(task="binary", random_state=7)
        b = LGBMAdapter(task="binary", random_state=7)
        a.fit(X, y)
        b.fit(X, y)
        np.testing.assert_array_equal(a.predict_proba(X), b.predict_proba(X))


# ---------------------------------------------------------------------------
# Multiclass classification
# ---------------------------------------------------------------------------


class TestLGBMMulticlass:
    def test_predict_proba_shape(
        self, multi_dataset: tuple[pd.DataFrame, pd.Series, int]
    ) -> None:
        X, y, n_classes = multi_dataset
        adapter = LGBMAdapter(task="multiclass", num_class=n_classes, random_state=0)
        adapter.fit(X, y)
        proba = adapter.predict_proba(X)
        assert proba.shape == (len(X), n_classes)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_predict_labels(
        self, multi_dataset: tuple[pd.DataFrame, pd.Series, int]
    ) -> None:
        X, y, n_classes = multi_dataset
        adapter = LGBMAdapter(task="multiclass", num_class=n_classes, random_state=0)
        adapter.fit(X, y)
        preds = adapter.predict(X)
        assert preds.shape == (len(X),)
        assert set(np.unique(preds)).issubset({0, 1, 2})
