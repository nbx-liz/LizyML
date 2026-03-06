"""Tests for Phase 15 — SHAP Explain.

Covers:
- compute_shap_values shape for regression / binary / multiclass
- OPTIONAL_DEP_MISSING when shap module is None
- predict(return_shap=True) E2E via Model
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

shap = pytest.importorskip("shap")  # noqa: E402

from lizyml import Model  # noqa: E402
from lizyml.core.exceptions import ErrorCode, LizyMLError  # noqa: E402
from lizyml.explain.shap_explainer import (  # noqa: E402
    compute_shap_values,
)

# ---------------------------------------------------------------------------
# Synthetic datasets (reused from e2e tests)
# ---------------------------------------------------------------------------


def _reg_df(n: int = 100, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "feat_a": rng.uniform(0, 10, n),
            "feat_b": rng.uniform(-1, 1, n),
        }
    )
    df["target"] = df["feat_a"] * 2.0 + df["feat_b"] + rng.normal(0, 0.1, n)
    return df


def _bin_df(n: int = 100, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "feat_a": rng.uniform(0, 10, n),
            "feat_b": rng.uniform(-1, 1, n),
        }
    )
    df["target"] = (df["feat_a"] > 5).astype(int)
    return df


def _multi_df(n: int = 150, seed: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "feat_a": rng.uniform(0, 10, n),
            "feat_b": rng.uniform(-1, 1, n),
        }
    )
    df["target"] = pd.cut(df["feat_a"], bins=3, labels=[0, 1, 2]).astype(int)
    return df


def _base_cfg(task: str, n_splits: int = 3) -> dict:
    return {
        "config_version": 1,
        "task": task,
        "data": {"target": "target"},
        "split": {"method": "kfold", "n_splits": n_splits, "random_state": 42},
        "model": {"name": "lgbm", "params": {"n_estimators": 10}},
        "training": {"seed": 0},
    }


# ---------------------------------------------------------------------------
# Helpers: fit and get transform X
# ---------------------------------------------------------------------------


def _fit_model(task: str) -> tuple[Model, pd.DataFrame]:
    """Return (fitted Model, X_new) for the given task."""
    if task == "regression":
        df = _reg_df()
    elif task == "binary":
        df = _bin_df()
    else:
        df = _multi_df()
    m = Model(_base_cfg(task))
    m.fit(data=df)
    X_new = df.drop(columns=["target"]).iloc[:5].reset_index(drop=True)
    return m, X_new


# ---------------------------------------------------------------------------
# OPTIONAL_DEP_MISSING
# ---------------------------------------------------------------------------


class TestOptionalDep:
    def test_raises_when_shap_none(self) -> None:
        df = _reg_df()
        m = Model(_base_cfg("regression"))
        m.fit(data=df)
        X_new = df.drop(columns=["target"]).iloc[:5]

        with (
            patch("lizyml.explain.shap_explainer._shap", None),
            pytest.raises(LizyMLError) as exc_info,
        ):
            m.predict(X_new, return_shap=True)
        assert exc_info.value.code == ErrorCode.OPTIONAL_DEP_MISSING
        assert "shap" in exc_info.value.context.get("package", "")

    def test_compute_shap_values_raises_when_shap_none(self) -> None:
        from lizyml.estimators.lgbm import LGBMAdapter

        adapter = LGBMAdapter(task="regression")
        X = pd.DataFrame({"a": [1.0]})
        with (
            patch("lizyml.explain.shap_explainer._shap", None),
            pytest.raises(LizyMLError) as exc_info,
        ):
            compute_shap_values(adapter, X, "regression")
        assert exc_info.value.code == ErrorCode.OPTIONAL_DEP_MISSING


# ---------------------------------------------------------------------------
# Shape contract — regression
# ---------------------------------------------------------------------------


class TestShapRegression:
    def test_shape(self) -> None:
        m, X_new = _fit_model("regression")
        result = m.predict(X_new, return_shap=True)
        assert result.shap_values is not None
        assert result.shap_values.shape == (5, 2)  # n_samples, n_features

    def test_none_when_not_requested(self) -> None:
        m, X_new = _fit_model("regression")
        result = m.predict(X_new)
        assert result.shap_values is None

    def test_pred_unchanged(self) -> None:
        m, X_new = _fit_model("regression")
        r1 = m.predict(X_new)
        r2 = m.predict(X_new, return_shap=True)
        np.testing.assert_array_almost_equal(r1.pred, r2.pred)


# ---------------------------------------------------------------------------
# Shape contract — binary
# ---------------------------------------------------------------------------


class TestShapBinary:
    def test_shape(self) -> None:
        m, X_new = _fit_model("binary")
        result = m.predict(X_new, return_shap=True)
        assert result.shap_values is not None
        assert result.shap_values.shape == (5, 2)

    def test_pred_unchanged(self) -> None:
        m, X_new = _fit_model("binary")
        r1 = m.predict(X_new)
        r2 = m.predict(X_new, return_shap=True)
        np.testing.assert_array_equal(r1.pred, r2.pred)


# ---------------------------------------------------------------------------
# Shape contract — multiclass
# ---------------------------------------------------------------------------


class TestShapMulticlass:
    def test_shape(self) -> None:
        m, X_new = _fit_model("multiclass")
        result = m.predict(X_new, return_shap=True)
        assert result.shap_values is not None
        # Always (n_samples, n_features) — classes averaged out
        assert result.shap_values.shape == (5, 2)

    def test_values_finite(self) -> None:
        """Reduced SHAP values are finite."""
        m, X_new = _fit_model("multiclass")
        result = m.predict(X_new, return_shap=True)
        assert result.shap_values is not None
        assert np.all(np.isfinite(result.shap_values))


# ---------------------------------------------------------------------------
# SHAP importance (H-0007)
# ---------------------------------------------------------------------------


class TestShapImportance:
    def test_regression_returns_dict(self) -> None:
        m, _ = _fit_model("regression")
        imp = m.importance(kind="shap")
        assert isinstance(imp, dict)
        assert "feat_a" in imp
        assert "feat_b" in imp
        assert all(isinstance(v, float) for v in imp.values())
        assert all(v >= 0 for v in imp.values())

    def test_binary_returns_dict(self) -> None:
        m, _ = _fit_model("binary")
        imp = m.importance(kind="shap")
        assert isinstance(imp, dict)
        assert len(imp) == 2

    def test_multiclass_returns_dict(self) -> None:
        m, _ = _fit_model("multiclass")
        imp = m.importance(kind="shap")
        assert isinstance(imp, dict)
        assert len(imp) == 2

    def test_before_fit_raises(self) -> None:
        m = Model(_base_cfg("regression"))
        with pytest.raises(LizyMLError) as exc_info:
            m.importance(kind="shap")
        assert exc_info.value.code == ErrorCode.MODEL_NOT_FIT

    def test_after_load_succeeds(self) -> None:
        import tempfile
        from pathlib import Path

        m, _ = _fit_model("regression")
        with tempfile.TemporaryDirectory() as td:
            export_dir = Path(td) / "model"
            m.export(str(export_dir))
            loaded = Model.load(str(export_dir))

        result = loaded.importance(kind="shap")
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_optional_dep_missing(self) -> None:
        m, _ = _fit_model("regression")
        with (
            patch("lizyml.explain.shap_explainer._shap", None),
            pytest.raises(LizyMLError) as exc_info,
        ):
            m.importance(kind="shap")
        assert exc_info.value.code == ErrorCode.OPTIONAL_DEP_MISSING

    def test_importance_plot_shap_returns_figure(self) -> None:
        import plotly.graph_objects as go

        m, _ = _fit_model("regression")
        fig = m.importance_plot(kind="shap")
        assert isinstance(fig, go.Figure)
