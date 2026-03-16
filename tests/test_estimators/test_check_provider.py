"""Category B: Provider/Adapter conformance suite (check_provider pattern).

Inspired by scikit-learn's check_estimator: a single parameterized suite
that auto-validates any EstimatorProvider + BaseEstimatorAdapter against
protocol invariants.  Adding a new provider → all checks run automatically.

See BLUEPRINT §18.1.5 and HISTORY H-0056 Category B.
"""

from __future__ import annotations

import pickle
from typing import Any

import numpy as np
import pytest

from lizyml.core.exceptions import ErrorCode, LizyMLError
from lizyml.core.types.search_dim import SearchDim
from lizyml.estimators.base import BaseEstimatorAdapter
from lizyml.estimators.lgbm.provider import LGBMProvider
from lizyml.features.pipeline_base import BaseFeaturePipeline
from tests._helpers import (
    make_binary_df,
    make_dense_float_20col,
    make_high_cardinality_cat_df,
    make_mixed_dtype_df,
    make_multiclass_df,
    make_regression_df,
    make_single_feature_df,
    make_with_missing_df,
)

# ---------------------------------------------------------------------------
# Provider registry — add new providers here; all checks run automatically
# ---------------------------------------------------------------------------

PROVIDERS: list[tuple[str, Any]] = [
    ("lgbm", LGBMProvider()),
]


def _make_model_cfg(provider_name: str) -> Any:
    """Build a minimal pydantic model config for the given provider."""
    if provider_name == "lgbm":
        from lizyml.config.schema import LGBMConfig

        return LGBMConfig(name="lgbm", params={"n_estimators": 5})
    msg = f"Unknown provider: {provider_name}"
    raise ValueError(msg)


def _make_data(task: str) -> tuple[Any, Any]:
    """Return (X, y) for the given task."""
    if task == "regression":
        df = make_regression_df(n=100, seed=42)
    elif task == "binary":
        df = make_binary_df(n=100, seed=42)
    else:
        df = make_multiclass_df(n=100, seed=42)
    X = df.drop(columns=["target"])
    y = df["target"]
    return X, y


def _build_and_fit(provider: Any, task: str, X: Any, y: Any) -> BaseEstimatorAdapter:
    """Build an adapter from the provider, fit it, and return."""
    n_classes = int(y.nunique()) if task == "multiclass" else None
    factory = provider.build_estimator_factory(
        task=task,
        params={"n_estimators": 5},
        n_classes=n_classes,
        early_stopping_rounds=None,
        seed=42,
    )
    adapter = factory()
    adapter.fit(X, y)
    return adapter


# ===================================================================
# Check 1: Protocol method return types
# ===================================================================


class TestProtocolReturnTypes:
    """Verify all protocol methods return expected types."""

    @pytest.mark.parametrize(
        "provider_name,provider", PROVIDERS, ids=[p[0] for p in PROVIDERS]
    )
    def test_extract_model_params_returns_dict(
        self, provider_name: str, provider: Any
    ) -> None:
        cfg = _make_model_cfg(provider_name)
        result = provider.extract_model_params(cfg)
        assert isinstance(result, dict)

    @pytest.mark.parametrize(
        "provider_name,provider", PROVIDERS, ids=[p[0] for p in PROVIDERS]
    )
    def test_extract_smart_params_returns_dict(
        self, provider_name: str, provider: Any
    ) -> None:
        cfg = _make_model_cfg(provider_name)
        result = provider.extract_smart_params(cfg)
        assert isinstance(result, dict)

    @pytest.mark.parametrize(
        "provider_name,provider", PROVIDERS, ids=[p[0] for p in PROVIDERS]
    )
    def test_runtime_deps_nonempty(self, provider_name: str, provider: Any) -> None:
        deps = provider.runtime_deps()
        assert isinstance(deps, dict)
        assert len(deps) > 0
        assert all(isinstance(v, str) and len(v) > 0 for v in deps.values())

    @pytest.mark.parametrize(
        "provider_name,provider", PROVIDERS, ids=[p[0] for p in PROVIDERS]
    )
    @pytest.mark.parametrize("task", ["regression", "binary", "multiclass"])
    def test_default_space_returns_search_dims(
        self, provider_name: str, provider: Any, task: str
    ) -> None:
        space = provider.default_space(task)
        assert isinstance(space, list)
        assert len(space) > 0
        assert all(isinstance(d, SearchDim) for d in space)
        assert all(hasattr(d, "name") for d in space)

    @pytest.mark.parametrize(
        "provider_name,provider", PROVIDERS, ids=[p[0] for p in PROVIDERS]
    )
    @pytest.mark.parametrize("task", ["regression", "binary", "multiclass"])
    def test_default_fixed_params_returns_dict(
        self, provider_name: str, provider: Any, task: str
    ) -> None:
        result = provider.default_fixed_params(task)
        assert isinstance(result, dict)


# ===================================================================
# Check 2: Factory → fit → predict roundtrip
# ===================================================================


class TestFactoryFitPredictRoundtrip:
    """Verify factory → fit → predict works for all task types."""

    @pytest.mark.parametrize(
        "provider_name,provider", PROVIDERS, ids=[p[0] for p in PROVIDERS]
    )
    @pytest.mark.parametrize("task", ["regression", "binary", "multiclass"])
    def test_fit_predict_roundtrip(
        self, provider_name: str, provider: Any, task: str
    ) -> None:
        X, y = _make_data(task)
        adapter = _build_and_fit(provider, task, X, y)
        preds = adapter.predict(X)
        assert preds.shape == (len(X),)

    @pytest.mark.parametrize(
        "provider_name,provider", PROVIDERS, ids=[p[0] for p in PROVIDERS]
    )
    def test_predict_proba_binary_shape(
        self, provider_name: str, provider: Any
    ) -> None:
        X, y = _make_data("binary")
        adapter = _build_and_fit(provider, "binary", X, y)
        proba = adapter.predict_proba(X)
        assert proba.shape == (len(X), 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    @pytest.mark.parametrize(
        "provider_name,provider", PROVIDERS, ids=[p[0] for p in PROVIDERS]
    )
    def test_predict_proba_multiclass_shape(
        self, provider_name: str, provider: Any
    ) -> None:
        X, y = _make_data("multiclass")
        n_classes = int(y.nunique())
        adapter = _build_and_fit(provider, "multiclass", X, y)
        proba = adapter.predict_proba(X)
        assert proba.shape == (len(X), n_classes)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    @pytest.mark.parametrize(
        "provider_name,provider", PROVIDERS, ids=[p[0] for p in PROVIDERS]
    )
    def test_predict_proba_regression_raises(
        self, provider_name: str, provider: Any
    ) -> None:
        X, y = _make_data("regression")
        adapter = _build_and_fit(provider, "regression", X, y)
        with pytest.raises(LizyMLError) as exc_info:
            adapter.predict_proba(X)
        assert exc_info.value.code == ErrorCode.UNSUPPORTED_TASK

    @pytest.mark.parametrize(
        "provider_name,provider", PROVIDERS, ids=[p[0] for p in PROVIDERS]
    )
    def test_predict_raw_shape(self, provider_name: str, provider: Any) -> None:
        X, y = _make_data("binary")
        adapter = _build_and_fit(provider, "binary", X, y)
        raw = adapter.predict_raw(X)
        assert raw.shape == (len(X),)


# ===================================================================
# Check 3: Pipeline factory
# ===================================================================


class TestPipelineFactory:
    """Verify build_pipeline_factory returns a valid pipeline."""

    @pytest.mark.parametrize(
        "provider_name,provider", PROVIDERS, ids=[p[0] for p in PROVIDERS]
    )
    def test_pipeline_factory_returns_pipeline(
        self, provider_name: str, provider: Any
    ) -> None:
        factory = provider.build_pipeline_factory()
        pipeline = factory()
        assert isinstance(pipeline, BaseFeaturePipeline)


# ===================================================================
# Check 4: Pickle roundtrip preserves predictions
# ===================================================================


class TestPickleRoundtrip:
    """Verify pickle → unpickle preserves exact predictions."""

    @pytest.mark.parametrize(
        "provider_name,provider", PROVIDERS, ids=[p[0] for p in PROVIDERS]
    )
    @pytest.mark.parametrize("task", ["regression", "binary", "multiclass"])
    def test_pickle_roundtrip(
        self, provider_name: str, provider: Any, task: str, tmp_path: Any
    ) -> None:
        X, y = _make_data(task)
        adapter = _build_and_fit(provider, task, X, y)
        preds_before = adapter.predict(X)

        pkl_path = tmp_path / "adapter.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(adapter, f)
        with open(pkl_path, "rb") as f:
            adapter_loaded = pickle.load(f)  # noqa: S301

        preds_after = adapter_loaded.predict(X)
        np.testing.assert_array_equal(preds_before, preds_after)


# ===================================================================
# Check 5: Importance after fit
# ===================================================================


class TestImportanceAfterFit:
    """Verify importance() returns correct keys after fit."""

    @pytest.mark.parametrize(
        "provider_name,provider", PROVIDERS, ids=[p[0] for p in PROVIDERS]
    )
    @pytest.mark.parametrize("kind", ["split", "gain"])
    def test_importance_keys_match_features(
        self, provider_name: str, provider: Any, kind: str
    ) -> None:
        X, y = _make_data("regression")
        adapter = _build_and_fit(provider, "regression", X, y)
        imp = adapter.importance(kind)  # type: ignore[arg-type]
        assert set(imp.keys()) == set(X.columns)
        assert all(isinstance(v, float) for v in imp.values())


# ===================================================================
# Check 6: Data diversity — fit/predict across diverse shapes
# ===================================================================

_DIVERSITY_MAKERS = [
    ("2col", make_regression_df),
    ("20col", make_dense_float_20col),
    ("mixed", make_mixed_dtype_df),
    ("missing", make_with_missing_df),
    ("single", make_single_feature_df),
    ("hi_card_cat", make_high_cardinality_cat_df),
]


class TestDataDiversity:
    """Verify fit/predict works across diverse data shapes."""

    @pytest.mark.parametrize(
        "provider_name,provider", PROVIDERS, ids=[p[0] for p in PROVIDERS]
    )
    @pytest.mark.parametrize(
        "data_name,make_df", _DIVERSITY_MAKERS, ids=[m[0] for m in _DIVERSITY_MAKERS]
    )
    def test_fit_predict_diverse_data(
        self,
        provider_name: str,
        provider: Any,
        data_name: str,
        make_df: Any,  # noqa: ARG002
    ) -> None:
        df = make_df(n=100, seed=42)
        X = df.drop(columns=["target"])
        y = df["target"]
        adapter = _build_and_fit(provider, "regression", X, y)
        preds = adapter.predict(X)
        assert preds.shape == (len(X),)
        assert np.all(np.isfinite(preds))


# ===================================================================
# Check 7: params_summary after fit
# ===================================================================


class TestParamsSummary:
    """Verify params_summary returns valid rows."""

    @pytest.mark.parametrize(
        "provider_name,provider", PROVIDERS, ids=[p[0] for p in PROVIDERS]
    )
    def test_params_summary_returns_list_of_dicts(
        self, provider_name: str, provider: Any
    ) -> None:
        X, y = _make_data("regression")
        cfg = _make_model_cfg(provider_name)
        adapter = _build_and_fit(provider, "regression", X, y)
        rows = provider.params_summary(adapter, cfg)
        assert isinstance(rows, list)
        assert len(rows) > 0
        assert all(isinstance(r, dict) for r in rows)
        assert all("parameter" in r and "value" in r for r in rows)
