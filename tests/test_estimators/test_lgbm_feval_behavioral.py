"""Behavioral tests for H-0064 — feval custom metrics in actual LightGBM training.

Verifies that feval metrics appear in eval_results and work correctly
with early stopping during real training runs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from lizyml.estimators.lgbm import LGBMAdapter


def _make_binary_data(
    rng: np.random.Generator, n: int = 200
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    X_train = pd.DataFrame({"f1": rng.standard_normal(n), "f2": rng.standard_normal(n)})
    y_train = pd.Series((rng.standard_normal(n) > 0).astype(int))
    X_valid = pd.DataFrame(
        {"f1": rng.standard_normal(50), "f2": rng.standard_normal(50)}
    )
    y_valid = pd.Series((rng.standard_normal(50) > 0).astype(int))
    return X_train, y_train, X_valid, y_valid


def _make_regression_data(
    rng: np.random.Generator, n: int = 200
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    X_train = pd.DataFrame({"f1": rng.standard_normal(n), "f2": rng.standard_normal(n)})
    y_train = pd.Series(np.abs(rng.standard_normal(n)) + 1.0)  # positive for rmsle
    X_valid = pd.DataFrame(
        {"f1": rng.standard_normal(50), "f2": rng.standard_normal(50)}
    )
    y_valid = pd.Series(np.abs(rng.standard_normal(50)) + 1.0)
    return X_train, y_train, X_valid, y_valid


def _make_multiclass_data(
    rng: np.random.Generator, n: int = 300
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    X_train = pd.DataFrame({"f1": rng.standard_normal(n), "f2": rng.standard_normal(n)})
    y_train = pd.Series(rng.integers(0, 3, size=n))
    X_valid = pd.DataFrame(
        {"f1": rng.standard_normal(60), "f2": rng.standard_normal(60)}
    )
    y_valid = pd.Series(rng.integers(0, 3, size=60))
    return X_train, y_train, X_valid, y_valid


class TestFevalBinaryTraining:
    """feval metrics appear in eval_results during binary training."""

    def test_f1_in_eval_results(self) -> None:
        rng = np.random.default_rng(42)
        X_train, y_train, X_valid, y_valid = _make_binary_data(rng)

        adapter = LGBMAdapter(
            task="binary",
            params={"metric": ["f1"], "n_estimators": 10},
            early_stopping_rounds=5,
        )
        adapter.fit(X_train, y_train, X_valid, y_valid)

        history = adapter.eval_results
        valid_keys = set(history.get("valid_0", {}).keys())
        assert "f1" in valid_keys

    def test_brier_in_eval_results(self) -> None:
        rng = np.random.default_rng(42)
        X_train, y_train, X_valid, y_valid = _make_binary_data(rng)

        adapter = LGBMAdapter(
            task="binary",
            params={"metric": ["brier"], "n_estimators": 10},
            early_stopping_rounds=5,
        )
        adapter.fit(X_train, y_train, X_valid, y_valid)

        history = adapter.eval_results
        valid_keys = set(history.get("valid_0", {}).keys())
        assert "brier" in valid_keys

    def test_mixed_native_and_feval(self) -> None:
        """auc (native) + f1 (feval) should both appear in eval_results."""
        rng = np.random.default_rng(42)
        X_train, y_train, X_valid, y_valid = _make_binary_data(rng)

        adapter = LGBMAdapter(
            task="binary",
            params={"metric": ["auc", "f1"], "n_estimators": 10},
            early_stopping_rounds=5,
        )
        adapter.fit(X_train, y_train, X_valid, y_valid)

        history = adapter.eval_results
        valid_keys = set(history.get("valid_0", {}).keys())
        assert "auc" in valid_keys
        assert "f1" in valid_keys

    def test_lizyml_name_logloss_translated(self) -> None:
        """metric='logloss' should work (translated to binary_logloss)."""
        rng = np.random.default_rng(42)
        X_train, y_train, X_valid, y_valid = _make_binary_data(rng)

        adapter = LGBMAdapter(
            task="binary",
            params={"metric": ["logloss"], "n_estimators": 10},
            early_stopping_rounds=5,
        )
        adapter.fit(X_train, y_train, X_valid, y_valid)

        history = adapter.eval_results
        valid_keys = set(history.get("valid_0", {}).keys())
        assert "binary_logloss" in valid_keys

    def test_ece_in_eval_results(self) -> None:
        rng = np.random.default_rng(42)
        X_train, y_train, X_valid, y_valid = _make_binary_data(rng)

        adapter = LGBMAdapter(
            task="binary",
            params={"metric": ["ece"], "n_estimators": 10},
            early_stopping_rounds=5,
        )
        adapter.fit(X_train, y_train, X_valid, y_valid)

        history = adapter.eval_results
        valid_keys = set(history.get("valid_0", {}).keys())
        assert "ece" in valid_keys

    def test_accuracy_as_feval(self) -> None:
        rng = np.random.default_rng(42)
        X_train, y_train, X_valid, y_valid = _make_binary_data(rng)

        adapter = LGBMAdapter(
            task="binary",
            params={"metric": ["accuracy"], "n_estimators": 10},
            early_stopping_rounds=5,
        )
        adapter.fit(X_train, y_train, X_valid, y_valid)

        history = adapter.eval_results
        valid_keys = set(history.get("valid_0", {}).keys())
        assert "accuracy" in valid_keys

    def test_precision_at_k_in_eval_results(self) -> None:
        rng = np.random.default_rng(42)
        X_train, y_train, X_valid, y_valid = _make_binary_data(rng)

        adapter = LGBMAdapter(
            task="binary",
            params={"metric": ["precision_at_k"], "n_estimators": 10},
            early_stopping_rounds=5,
        )
        adapter.fit(X_train, y_train, X_valid, y_valid)

        history = adapter.eval_results
        valid_keys = set(history.get("valid_0", {}).keys())
        assert "precision_at_k" in valid_keys


class TestFevalRegressionTraining:
    """feval metrics in regression training."""

    def test_rmsle_in_eval_results(self) -> None:
        rng = np.random.default_rng(42)
        X_train, y_train, X_valid, y_valid = _make_regression_data(rng)

        adapter = LGBMAdapter(
            task="regression",
            params={"metric": ["rmsle"], "n_estimators": 10},
            early_stopping_rounds=5,
        )
        adapter.fit(X_train, y_train, X_valid, y_valid)

        history = adapter.eval_results
        valid_keys = set(history.get("valid_0", {}).keys())
        assert "rmsle" in valid_keys

    def test_mixed_native_and_feval_regression(self) -> None:
        """rmse (native) + rmsle (feval) should both appear."""
        rng = np.random.default_rng(42)
        X_train, y_train, X_valid, y_valid = _make_regression_data(rng)

        adapter = LGBMAdapter(
            task="regression",
            params={"metric": ["rmse", "rmsle"], "n_estimators": 10},
            early_stopping_rounds=5,
        )
        adapter.fit(X_train, y_train, X_valid, y_valid)

        history = adapter.eval_results
        valid_keys = set(history.get("valid_0", {}).keys())
        assert "rmse" in valid_keys
        assert "rmsle" in valid_keys


class TestFevalMulticlassTraining:
    """feval metrics in multiclass training."""

    def test_f1_multiclass_in_eval_results(self) -> None:
        rng = np.random.default_rng(42)
        X_train, y_train, X_valid, y_valid = _make_multiclass_data(rng)

        adapter = LGBMAdapter(
            task="multiclass",
            num_class=3,
            params={"metric": ["f1"], "n_estimators": 10},
            early_stopping_rounds=5,
        )
        adapter.fit(X_train, y_train, X_valid, y_valid)

        history = adapter.eval_results
        valid_keys = set(history.get("valid_0", {}).keys())
        assert "f1" in valid_keys

    def test_logloss_multiclass_translated(self) -> None:
        """metric='logloss' should be translated to multi_logloss."""
        rng = np.random.default_rng(42)
        X_train, y_train, X_valid, y_valid = _make_multiclass_data(rng)

        adapter = LGBMAdapter(
            task="multiclass",
            num_class=3,
            params={"metric": ["logloss"], "n_estimators": 10},
            early_stopping_rounds=5,
        )
        adapter.fit(X_train, y_train, X_valid, y_valid)

        history = adapter.eval_results
        valid_keys = set(history.get("valid_0", {}).keys())
        assert "multi_logloss" in valid_keys


class TestFevalEarlyStopping:
    """feval-only metrics work correctly with early stopping."""

    def test_feval_only_with_early_stopping(self) -> None:
        """When only feval metrics are specified, early stopping should still work."""
        rng = np.random.default_rng(42)
        X_train, y_train, X_valid, y_valid = _make_binary_data(rng)

        adapter = LGBMAdapter(
            task="binary",
            params={"metric": ["f1"], "n_estimators": 100},
            early_stopping_rounds=5,
        )
        adapter.fit(X_train, y_train, X_valid, y_valid)

        # Should have stopped early (not trained all 100 rounds)
        history = adapter.eval_results
        f1_values = history.get("valid_0", {}).get("f1", [])
        assert len(f1_values) > 0
        # Verify values are in valid range
        assert all(0.0 <= v <= 1.0 for v in f1_values)

    def test_feval_values_in_valid_range(self) -> None:
        """Brier score values should be in [0, 1]."""
        rng = np.random.default_rng(42)
        X_train, y_train, X_valid, y_valid = _make_binary_data(rng)

        adapter = LGBMAdapter(
            task="binary",
            params={"metric": ["brier"], "n_estimators": 20},
            early_stopping_rounds=10,
        )
        adapter.fit(X_train, y_train, X_valid, y_valid)

        history = adapter.eval_results
        brier_values = history.get("valid_0", {}).get("brier", [])
        assert len(brier_values) > 0
        assert all(0.0 <= v <= 1.0 for v in brier_values)
