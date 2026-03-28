"""Integration tests for H-0065 MetricEntry across the full stack.

Tests metric_bridge feval display names, _build_params dict metric support,
params_summary display, and EvaluationConfig dict form.
"""

from __future__ import annotations

import pytest

from lizyml.config.schema import EvaluationConfig
from lizyml.estimators.lgbm.adapter import LGBMAdapter
from lizyml.estimators.lgbm.metric_bridge import (
    _metric_display_name,
    resolve_metrics,
)
from lizyml.metrics.classification import PrecisionAtK


class TestMetricDisplayName:
    """_metric_display_name builds human-readable names."""

    def test_no_kwargs(self) -> None:
        metric = PrecisionAtK()
        assert _metric_display_name(metric, {}) == "precision_at_k"

    def test_with_k(self) -> None:
        metric = PrecisionAtK(k=20)
        assert _metric_display_name(metric, {"k": 20}) == "precision_at_k (k=20)"


class TestResolveMetricsWithDict:
    """resolve_metrics handles MetricEntry dicts (H-0065)."""

    def test_str_backward_compat(self) -> None:
        native, fevals, display_names = resolve_metrics(
            ["auc", "precision_at_k"], "binary"
        )
        assert native == ["auc"]
        assert len(fevals) == 1
        # Default k → no params in display name
        assert display_names == ["precision_at_k"]

    def test_dict_entry_custom_k(self) -> None:
        native, fevals, display_names = resolve_metrics(
            [{"precision_at_k": {"k": 25}}], "binary"
        )
        assert native == []
        assert len(fevals) == 1
        assert display_names == ["precision_at_k (k=25)"]

    def test_mixed_entries(self) -> None:
        native, fevals, display_names = resolve_metrics(
            ["auc", {"precision_at_k": {"k": 5}}], "binary"
        )
        assert native == ["auc"]
        assert len(fevals) == 1
        assert display_names == ["precision_at_k (k=5)"]

    def test_duplicate_name_different_k_rejected(self) -> None:
        """Same metric name with different k values is rejected as duplicate."""
        from lizyml.core.exceptions import LizyMLError

        with pytest.raises(LizyMLError, match="Duplicate"):
            resolve_metrics(
                [{"precision_at_k": {"k": 5}}, {"precision_at_k": {"k": 20}}],
                "binary",
            )

    def test_unknown_dict_metric_rejected(self) -> None:
        """Dict entry with unknown metric name is rejected."""
        from lizyml.core.exceptions import LizyMLError

        with pytest.raises(LizyMLError):
            resolve_metrics([{"totally_unknown": {"k": 5}}], "binary")

    def test_non_parameterised_metric_with_empty_dict(self) -> None:
        """Dict entry for non-parameterised metric with empty kwargs works."""
        native, fevals, display_names = resolve_metrics([{"auc": {}}], "binary")
        assert native == ["auc"]
        assert fevals == []

    def test_feval_returns_display_name(self) -> None:
        """The feval callable should return the display name, not bare metric name."""
        import lightgbm as lgb
        import numpy as np

        _, fevals, _ = resolve_metrics([{"precision_at_k": {"k": 30}}], "binary")
        feval_fn = fevals[0]

        # Create minimal dataset
        y = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=np.float64)
        # Binary: raw logits (will be passed through sigmoid)
        preds = np.array([2.0, -2.0, 1.5, -1.5, 1.0, -1.0, 0.5, -0.5, 0.0, 0.0])
        ds = lgb.Dataset(np.zeros((10, 1)), label=y)
        ds.construct()

        name, value, is_higher = feval_fn(preds, ds)
        assert name == "precision_at_k (k=30)"
        assert is_higher is True
        assert isinstance(value, float)


class TestBuildParamsWithDictMetric:
    """_build_params handles dict-form metric entries (H-0065)."""

    def test_dict_metric_single(self) -> None:
        adapter = LGBMAdapter(
            task="binary",
            params={"metric": {"precision_at_k": {"k": 15}}},
        )
        params, _, feval_list, display_names = adapter._build_params()
        # precision_at_k is feval-only → native metric should be "None"
        assert params["metric"] == "None"
        assert len(feval_list) == 1
        assert display_names == ["precision_at_k (k=15)"]

    def test_dict_metric_in_list(self) -> None:
        adapter = LGBMAdapter(
            task="binary",
            params={"metric": ["auc", {"precision_at_k": {"k": 20}}]},
        )
        params, _, feval_list, display_names = adapter._build_params()
        assert params["metric"] == ["auc"]
        assert len(feval_list) == 1
        assert display_names == ["precision_at_k (k=20)"]

    def test_str_metric_backward_compat(self) -> None:
        adapter = LGBMAdapter(
            task="binary",
            params={"metric": "auc"},
        )
        params, _, feval_list, display_names = adapter._build_params()
        assert params["metric"] == ["auc"]
        assert feval_list == []
        assert display_names == []


class TestEvaluationConfigDictMetrics:
    """EvaluationConfig accepts dict-form metric entries."""

    def test_str_only(self) -> None:
        cfg = EvaluationConfig(metrics=["rmse", "mae"])
        assert cfg.metrics == ["rmse", "mae"]

    def test_dict_entry(self) -> None:
        cfg = EvaluationConfig(metrics=["auc", {"precision_at_k": {"k": 20}}])
        assert len(cfg.metrics) == 2
        assert cfg.metrics[0] == "auc"
        assert cfg.metrics[1] == {"precision_at_k": {"k": 20}}

    def test_empty_default(self) -> None:
        cfg = EvaluationConfig()
        assert cfg.metrics == []

    def test_extra_forbid(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            EvaluationConfig(metrics=["auc"], unknown_field="x")  # type: ignore[call-arg]


class TestParamsSummaryFevalDisplay:
    """params_summary shows feval display names after fit (H-0065)."""

    def test_feval_display_names_stored(self) -> None:
        adapter = LGBMAdapter(
            task="binary",
            params={"metric": [{"precision_at_k": {"k": 20}}]},
        )
        # Before fit, display names are empty
        assert adapter._feval_display_names == []
        # After _build_params, they are populated
        _, _, _, display_names = adapter._build_params()
        assert display_names == ["precision_at_k (k=20)"]
